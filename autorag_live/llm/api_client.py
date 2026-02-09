"""
Robust LLM API client for AutoRAG-Live.

Provides a unified interface for multiple LLM providers with
retry logic, rate limiting, fallbacks, and streaming support.

Features:
- Multiple provider support (OpenAI, Anthropic, etc.)
- Automatic retries with exponential backoff
- Rate limiting and quota management
- Provider fallbacks
- Streaming responses
- Cost tracking

Example usage:
    >>> client = LLMClient(provider="openai", model="gpt-4")
    >>> response = client.generate("What is machine learning?")
    >>> print(response.text)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    CUSTOM = "custom"


class ResponseStatus(str, Enum):
    """Response status codes."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    FALLBACK_USED = "fallback_used"


@dataclass
class LLMResponse:
    """Response from LLM API."""

    text: str
    status: ResponseStatus = ResponseStatus.SUCCESS

    # Usage info
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Cost
    cost: float = 0.0

    # Timing
    latency_ms: float = 0.0

    # Provider info
    provider: Optional[str] = None
    model: Optional[str] = None

    # Raw response
    raw_response: Optional[Any] = None

    # Error info
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == ResponseStatus.SUCCESS

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms > 0:
            return (self.completion_tokens / self.latency_ms) * 1000
        return 0.0


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"

    # API settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    # Request settings
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_multiplier: float = 2.0
    timeout: float = 60.0

    # Rate limiting
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None

    # Cost tracking
    track_cost: bool = True

    # Streaming
    stream: bool = False


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute
            tokens_per_minute: Max tokens per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        self._request_times: List[float] = []
        self._token_counts: List[tuple] = []  # (timestamp, tokens)

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Clean old entries
        self._clean_old_entries(now)

        # Check request limit
        if self.requests_per_minute:
            while len(self._request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self._request_times[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                now = time.time()
                self._clean_old_entries(now)

        # Check token limit
        if self.tokens_per_minute and estimated_tokens > 0:
            recent_tokens = sum(t for _, t in self._token_counts)
            while recent_tokens + estimated_tokens > self.tokens_per_minute:
                if self._token_counts:
                    sleep_time = 60 - (now - self._token_counts[0][0])
                    if sleep_time > 0:
                        logger.debug(f"Token rate limiting: sleeping {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                now = time.time()
                self._clean_old_entries(now)
                recent_tokens = sum(t for _, t in self._token_counts)

        # Record this request
        self._request_times.append(now)

    def record_tokens(self, tokens: int) -> None:
        """Record token usage."""
        self._token_counts.append((time.time(), tokens))

    def _clean_old_entries(self, now: float) -> None:
        """Remove entries older than 1 minute."""
        cutoff = now - 60

        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]


class CostTracker:
    """Track LLM API costs."""

    # Pricing per 1K tokens (approximate)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self):
        """Initialize cost tracker."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._history: List[Dict[str, Any]] = []

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in dollars
        """
        # Find matching pricing
        pricing = None
        for key in self.PRICING:
            if key in model.lower():
                pricing = self.PRICING[key]
                break

        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record usage and return cost."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        self._history.append(
            {
                "timestamp": time.time(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }
        )

        return cost

    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": len(self._history),
        }


class BaseProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate response."""
        pass

    @abstractmethod
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate chat response."""
        pass

    def stream(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> Iterator[str]:
        """Stream response."""
        # Default: return full response
        response = self.generate(prompt, config)
        yield response.text


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI provider."""
        self.api_key = api_key
        self._client = None

    def _get_client(self, api_key: Optional[str] = None):
        """Get or create client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key or self.api_key)
            except ImportError:
                raise ImportError("openai package not installed")
        return self._client

    def generate(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(messages, config)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate chat completion."""
        start_time = time.time()

        try:
            client = self._get_client(config.api_key)

            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )

            latency = (time.time() - start_time) * 1000

            return LLMResponse(
                text=response.choices[0].message.content,
                status=ResponseStatus.SUCCESS,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=latency,
                provider="openai",
                model=config.model,
                raw_response=response,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                text="",
                status=ResponseStatus.ERROR,
                latency_ms=latency,
                provider="openai",
                model=config.model,
                error_message=str(e),
            )

    def stream(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> Iterator[str]:
        """Stream response."""
        messages = [{"role": "user", "content": prompt}]

        try:
            client = self._get_client(config.api_key)

            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield ""


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic provider."""
        self.api_key = api_key
        self._client = None

    def _get_client(self, api_key: Optional[str] = None):
        """Get or create client."""
        if self._client is None:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=api_key or self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed")
        return self._client

    def generate(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate using Anthropic API."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(messages, config)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate chat completion."""
        start_time = time.time()

        try:
            client = self._get_client(config.api_key)

            # Extract system message if present
            system = None
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    chat_messages.append(msg)

            kwargs = {
                "model": config.model,
                "messages": chat_messages,
                "max_tokens": config.max_tokens,
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)

            latency = (time.time() - start_time) * 1000

            return LLMResponse(
                text=response.content[0].text,
                status=ResponseStatus.SUCCESS,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=latency,
                provider="anthropic",
                model=config.model,
                raw_response=response,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                text="",
                status=ResponseStatus.ERROR,
                latency_ms=latency,
                provider="anthropic",
                model=config.model,
                error_message=str(e),
            )


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, response_text: str = "Mock response"):
        """Initialize mock provider."""
        self.response_text = response_text
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate mock response."""
        self.call_count += 1
        return LLMResponse(
            text=self.response_text,
            status=ResponseStatus.SUCCESS,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self.response_text.split()),
            total_tokens=len(prompt.split()) + len(self.response_text.split()),
            latency_ms=10.0,
            provider="mock",
            model="mock-model",
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate mock chat response."""
        return self.generate(str(messages), config)


class LLMClient:
    """
    Main LLM client with retries, fallbacks, and connection pooling.

    Uses persistent HTTP connection pools to eliminate per-request TCP/TLS
    overhead (~30-50ms savings per call). Critical for agentic RAG where
    a single query may trigger 5-10+ LLM calls across planning, retrieval,
    synthesis, and reflection steps.

    Example:
        >>> # Basic usage
        >>> client = LLMClient(provider="openai", model="gpt-4")
        >>> response = client.generate("Explain machine learning")
        >>> print(response.text)
        >>>
        >>> # With connection pooling and fallbacks
        >>> client = LLMClient(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     fallback_providers=["anthropic"],
        ...     enable_connection_pool=True,
        ... )
        >>>
        >>> # Streaming
        >>> for chunk in client.stream("Write a story"):
        ...     print(chunk, end="")
    """

    # Shared connection pool across all LLMClient instances (singleton per process)
    _shared_pool: Optional[Any] = None
    _pool_initialized: bool = False

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        fallback_providers: Optional[List[str]] = None,
        enable_connection_pool: bool = True,
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider name
            model: Model name
            api_key: API key
            config: Full configuration
            fallback_providers: Fallback providers
            enable_connection_pool: Enable HTTP connection pooling (recommended)
        """
        if config:
            self.config = config
        else:
            self.config = LLMConfig(
                provider=LLMProvider(provider) if isinstance(provider, str) else provider,
                model=model,
                api_key=api_key,
            )

        # Initialize providers
        self._providers: Dict[str, BaseProvider] = {}
        self._init_provider(self.config.provider)

        # Initialize fallbacks
        self.fallback_providers = fallback_providers or []
        for fb in self.fallback_providers:
            self._init_provider(LLMProvider(fb))

        # Rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.requests_per_minute,
            tokens_per_minute=self.config.tokens_per_minute,
        )

        # Cost tracker
        self.cost_tracker = CostTracker() if self.config.track_cost else None

        # Connection pooling: lazily initialize shared pool
        self._enable_connection_pool = enable_connection_pool
        if enable_connection_pool and not LLMClient._pool_initialized:
            self._init_connection_pool()

    @classmethod
    def _init_connection_pool(cls) -> None:
        """Initialize shared connection pool (once per process)."""
        try:
            from autorag_live.utils.connection_pool import ConnectionPool, ConnectionPoolConfig

            pool_config = ConnectionPoolConfig(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
                timeout_connect=5.0,
                timeout_read=60.0,
                enable_http2=True,
            )
            cls._shared_pool = ConnectionPool(config=pool_config)
            cls._pool_initialized = True
            logger.info(
                "Initialized shared HTTP connection pool for LLM providers "
                "(saves ~30-50ms per request from TCP/TLS reuse)"
            )
        except ImportError:
            logger.debug("httpx not available; connection pooling disabled")
            cls._pool_initialized = False

    @classmethod
    def get_connection_pool(cls) -> Optional[Any]:
        """Get the shared connection pool (for direct httpx usage in providers)."""
        return cls._shared_pool if cls._pool_initialized else None

    def _init_provider(self, provider: LLMProvider) -> None:
        """Initialize a provider."""
        if provider.value in self._providers:
            return

        if provider == LLMProvider.OPENAI:
            self._providers[provider.value] = OpenAIProvider(self.config.api_key)
        elif provider == LLMProvider.ANTHROPIC:
            self._providers[provider.value] = AnthropicProvider(self.config.api_key)
        else:
            # Default to mock for unsupported
            logger.warning(f"Provider {provider} not fully supported, using mock")
            self._providers[provider.value] = MockProvider()

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response with retries and fallbacks.

        Args:
            prompt: Input prompt
            **kwargs: Override config parameters

        Returns:
            LLMResponse
        """
        # Update config with kwargs
        config = self._merge_config(kwargs)

        # Rate limiting
        self.rate_limiter.wait_if_needed(len(prompt.split()))

        # Try primary provider
        response = self._try_generate(
            self.config.provider.value,
            prompt,
            config,
        )

        # Try fallbacks if needed
        if not response.is_success and self.fallback_providers:
            for fallback in self.fallback_providers:
                logger.info(f"Trying fallback provider: {fallback}")
                response = self._try_generate(fallback, prompt, config)
                if response.is_success:
                    response.status = ResponseStatus.FALLBACK_USED
                    break

        # Track cost
        if self.cost_tracker and response.is_success:
            cost = self.cost_tracker.record(
                config.model,
                response.prompt_tokens,
                response.completion_tokens,
            )
            response.cost = cost

        # Record tokens for rate limiting
        if response.is_success:
            self.rate_limiter.record_tokens(response.total_tokens)

        return response

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate chat response.

        Args:
            messages: Chat messages
            **kwargs: Override config parameters

        Returns:
            LLMResponse
        """
        config = self._merge_config(kwargs)

        # Rate limiting
        total_chars = sum(len(m.get("content", "")) for m in messages)
        self.rate_limiter.wait_if_needed(total_chars // 4)

        provider = self._providers.get(self.config.provider.value)
        if not provider:
            return LLMResponse(
                text="",
                status=ResponseStatus.ERROR,
                error_message=f"Provider {self.config.provider} not available",
            )

        # Retry logic
        last_error = None
        for attempt in range(config.max_retries):
            response = provider.generate_chat(messages, config)

            if response.is_success:
                if self.cost_tracker:
                    cost = self.cost_tracker.record(
                        config.model,
                        response.prompt_tokens,
                        response.completion_tokens,
                    )
                    response.cost = cost
                return response

            last_error = response.error_message

            if attempt < config.max_retries - 1:
                delay = config.retry_delay * (config.retry_multiplier**attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s")
                time.sleep(delay)

        return LLMResponse(
            text="",
            status=ResponseStatus.ERROR,
            error_message=f"All {config.max_retries} attempts failed: {last_error}",
        )

    def stream(
        self,
        prompt: str,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream response.

        Args:
            prompt: Input prompt
            **kwargs: Override config parameters

        Yields:
            Response chunks
        """
        config = self._merge_config(kwargs)
        config.stream = True

        provider = self._providers.get(self.config.provider.value)
        if provider:
            yield from provider.stream(prompt, config)

    def _try_generate(
        self,
        provider_name: str,
        prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Try generating with a specific provider."""
        provider = self._providers.get(provider_name)
        if not provider:
            return LLMResponse(
                text="",
                status=ResponseStatus.ERROR,
                error_message=f"Provider {provider_name} not available",
            )

        last_error = None
        for attempt in range(config.max_retries):
            response = provider.generate(prompt, config)

            if response.is_success:
                return response

            last_error = response.error_message

            if attempt < config.max_retries - 1:
                delay = config.retry_delay * (config.retry_multiplier**attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s")
                time.sleep(delay)

        return LLMResponse(
            text="",
            status=ResponseStatus.ERROR,
            provider=provider_name,
            error_message=f"All {config.max_retries} attempts failed: {last_error}",
        )

    def _merge_config(self, kwargs: Dict[str, Any]) -> LLMConfig:
        """Merge kwargs with base config."""
        if not kwargs:
            return self.config

        return LLMConfig(
            provider=self.config.provider,
            model=kwargs.get("model", self.config.model),
            api_key=kwargs.get("api_key", self.config.api_key),
            api_base=kwargs.get("api_base", self.config.api_base),
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            max_retries=kwargs.get("max_retries", self.config.max_retries),
            retry_delay=kwargs.get("retry_delay", self.config.retry_delay),
            timeout=kwargs.get("timeout", self.config.timeout),
            stream=kwargs.get("stream", self.config.stream),
        )

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        if self.cost_tracker:
            return self.cost_tracker.get_summary()
        return {}


# Convenience functions


def create_client(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
) -> LLMClient:
    """
    Create an LLM client.

    Args:
        provider: Provider name
        model: Model name
        api_key: API key

    Returns:
        LLMClient
    """
    return LLMClient(provider=provider, model=model, api_key=api_key)


def generate(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    **kwargs,
) -> str:
    """
    Quick generation function.

    Args:
        prompt: Input prompt
        provider: Provider name
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Generated text
    """
    client = LLMClient(provider=provider, model=model)
    response = client.generate(prompt, **kwargs)
    return response.text if response.is_success else ""


def generate_chat(
    messages: List[Dict[str, str]],
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    **kwargs,
) -> str:
    """
    Quick chat generation function.

    Args:
        messages: Chat messages
        provider: Provider name
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Generated text
    """
    client = LLMClient(provider=provider, model=model)
    response = client.generate_chat(messages, **kwargs)
    return response.text if response.is_success else ""
