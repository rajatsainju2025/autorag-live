from __future__ import annotations

from typing import Any, List

from autorag_live.core.interfaces import LLMProvider


class MockLLM(LLMProvider):
    """A minimal mock LLM provider for tests and local development."""

    def __init__(self, prefix: str = "mock:") -> None:
        self.prefix = prefix

    async def agenerate(self, prompts: List[str]) -> Any:
        return [self._sync_invoke(p) for p in prompts]

    async def ainvoke(self, prompt: str) -> Any:
        return self._sync_invoke(prompt)

    def invoke(self, prompt: str) -> Any:
        return self._sync_invoke(prompt)

    def _sync_invoke(self, prompt: str) -> str:
        return f"{self.prefix} {prompt}"


class LLMFactory:
    """Simple factory to provide an LLM provider instance by name."""

    registry: dict[str, type[LLMProvider]] = {"mock": MockLLM}

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        cls.registry[name] = provider_cls

    @classmethod
    def create(cls, name: str = "mock", **kwargs) -> LLMProvider:
        provider_cls = cls.registry.get(name)
        if provider_cls is None:
            raise ValueError(f"Unknown LLM provider: {name}")
        return provider_cls(**kwargs)
