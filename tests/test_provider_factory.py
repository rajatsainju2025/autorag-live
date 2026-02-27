"""Tests for LLM provider factory."""

import pytest

from autorag_live.core.protocols import GenerationResult, Message
from autorag_live.llm.provider_factory import (
    BaseLLMProvider,
    LLMProviderFactory,
    MockLLMProvider,
    ProviderConfig,
)


class TestMockLLMProvider:
    @pytest.mark.asyncio
    async def test_default_response(self):
        llm = MockLLMProvider()
        result = await llm.generate([Message.user("Hello")])

        assert isinstance(result, GenerationResult)
        assert result.content == "This is a mock LLM response."
        assert result.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_custom_responses(self):
        llm = MockLLMProvider(
            responses={
                "What is RLHF?": "RLHF stands for Reinforcement Learning from Human Feedback."
            }
        )
        result = await llm.generate([Message.user("What is RLHF?")])
        assert "Reinforcement Learning" in result.content

    @pytest.mark.asyncio
    async def test_call_history_tracked(self):
        llm = MockLLMProvider()
        await llm.generate([Message.user("q1")])
        await llm.generate([Message.user("q2")])

        assert len(llm.call_history) == 2
        assert llm.stats["calls"] == 2
        assert llm.stats["total_tokens"] == 60

    @pytest.mark.asyncio
    async def test_system_message_preserved(self):
        llm = MockLLMProvider()
        msgs = [Message.system("You are helpful"), Message.user("Hi")]
        result = await llm.generate(msgs)
        assert result.content == "This is a mock LLM response."


class TestLLMProviderFactory:
    def test_create_mock(self):
        llm = LLMProviderFactory.create("mock")
        assert isinstance(llm, MockLLMProvider)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.create("nonexistent")

    def test_available_providers(self):
        providers = LLMProviderFactory.available_providers()
        assert "mock" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers

    def test_create_with_config(self):
        config = ProviderConfig(model="test-model", temperature=0.1)
        llm = LLMProviderFactory.create("mock", config=config)
        assert llm.config.model == "test-model"
        assert llm.config.temperature == 0.1

    def test_register_custom_provider(self):
        class CustomProvider(BaseLLMProvider):
            pass

        LLMProviderFactory.register("custom", CustomProvider)
        assert "custom" in LLMProviderFactory.available_providers()
        llm = LLMProviderFactory.create("custom", config=ProviderConfig())
        assert isinstance(llm, CustomProvider)

    def test_case_insensitive(self):
        llm = LLMProviderFactory.create("Mock")
        assert isinstance(llm, MockLLMProvider)


class TestProviderConfig:
    def test_defaults(self):
        cfg = ProviderConfig()
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048

    def test_custom_values(self):
        cfg = ProviderConfig(model="claude-sonnet-4-20250514", temperature=0.0, api_key="sk-test")
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.api_key == "sk-test"


class TestProviderStats:
    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        llm = MockLLMProvider()
        assert llm.stats["calls"] == 0

        await llm.generate([Message.user("test")])
        assert llm.stats["calls"] == 1
        assert llm.stats["total_tokens"] == 30
        assert llm.stats["provider"] == "MockLLMProvider"
