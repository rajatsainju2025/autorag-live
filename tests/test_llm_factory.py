
from autorag_live.llm.factory import LLMFactory, MockLLM


async def test_mock_llm_invoke():
    m = MockLLM(prefix="test")
    out = m.invoke("hello")
    assert "test" in out


async def test_mock_llm_ainvoke():
    m = MockLLM(prefix="async")
    out = await m.ainvoke("world")
    assert "async" in out


def test_factory_create():
    inst = LLMFactory.create("mock", prefix="factory")
    assert isinstance(inst, MockLLM)
