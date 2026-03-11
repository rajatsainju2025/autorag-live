
import pytest

from autorag_live.llm.adapters import AsyncLLMAdapter
from autorag_live.llm.factory import MockLLM


@pytest.mark.asyncio
async def test_async_llm_adapter_ainvoke():
    sync_llm = MockLLM(prefix="sync")
    adapter = AsyncLLMAdapter(sync_llm)
    out = await adapter.ainvoke("hello")
    assert "sync" in out


@pytest.mark.asyncio
async def test_async_llm_adapter_agenerate():
    sync_llm = MockLLM(prefix="batch")
    adapter = AsyncLLMAdapter(sync_llm)
    out = await adapter.agenerate(["a", "b"])  # type: ignore
    assert isinstance(out, list)
    assert any("batch" in o for o in out)


def test_invoke_passthrough():
    sync_llm = MockLLM(prefix="direct")
    adapter = AsyncLLMAdapter(sync_llm)
    out = adapter.invoke("x")
    assert "direct" in out
