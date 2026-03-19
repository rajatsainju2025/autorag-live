import pytest

from autorag_live.knowledge_graph.entity_extractor import EntityExtractor


class FakeGen:
    def __init__(self, text: str):
        self.text = text


class FakeLLM:
    def __init__(self, text: str):
        self._text = text

    async def agenerate(self, prompts):
        # emulate the shape expected by EntityExtractor
        class Resp:
            def __init__(self, text):
                self.generations = [[FakeGen(text)]]

        return Resp(self._text)


class FakeAInvokeLLM:
    def __init__(self, text: str):
        self._text = text

    async def ainvoke(self, prompt):
        class Resp:
            def __init__(self, text):
                self.content = text

        return Resp(self._text)


class FakeInvokeLLM:
    def __init__(self, text: str):
        self._text = text

    def invoke(self, prompt):
        class Resp:
            def __init__(self, text):
                self.content = text

        return Resp(self._text)


@pytest.mark.asyncio
async def test_extract_empty_returns_empty():
    llm = FakeLLM("")
    extractor = EntityExtractor(llm)
    out = await extractor.extract("")
    assert out["entities"] == [] and out["relationships"] == []


@pytest.mark.asyncio
async def test_extract_parses_json_response():
    json_text = '{"entities":[{"id":"Alice","type":"PERSON","description":"A person"}],"relationships":[{"source":"Alice","target":"Bob","type":"KNOWS","description":"knows"}]}'
    llm = FakeLLM(json_text)
    extractor = EntityExtractor(llm)
    out = await extractor.extract("Some text")
    assert isinstance(out, dict)
    assert len(out.get("entities", [])) == 1
    assert len(out.get("relationships", [])) == 1


@pytest.mark.asyncio
async def test_extract_supports_ainvoke():
    llm = FakeAInvokeLLM('{"entities":[],"relationships":[]}')
    extractor = EntityExtractor(llm)

    out = await extractor.extract("Some text")

    assert out["entities"] == []
    assert out["relationships"] == []


@pytest.mark.asyncio
async def test_extract_supports_sync_invoke():
    llm = FakeInvokeLLM('{"entities":[],"relationships":[]}')
    extractor = EntityExtractor(llm)

    out = await extractor.extract("Some text")

    assert out["entities"] == []
    assert out["relationships"] == []


@pytest.mark.asyncio
async def test_process_documents_attaches_knowledge_graph():
    llm = FakeLLM(
        '{"entities":[{"id":"Alice","type":"PERSON","description":"A person"}],"relationships":[]}'
    )
    extractor = EntityExtractor(llm)
    documents = [{"text": "Alice text"}, {"text": ""}]

    out = await extractor.process_documents(documents)

    assert out[0]["knowledge_graph"]["entities"]
    assert out[1]["knowledge_graph"]["entities"] == []
