import pytest

from autorag_live.knowledge_graph.graph_rag import (
    EntityType,
    LLMEntityExtractor,
    RuleBasedEntityExtractor,
)


@pytest.mark.asyncio
async def test_rule_based_extractor_deduplicates_capitalized_phrases():
    extractor = RuleBasedEntityExtractor()
    text = "OpenAI launched ChatGPT. OpenAI is based in San Francisco."

    entities = await extractor.extract(text)
    names = [entity.name for entity in entities]

    assert names.count("OpenAI") == 1
    assert names.count("San Francisco") == 1


def test_llm_entity_extractor_uses_constant_type_lookup():
    extractor = LLMEntityExtractor(llm=lambda prompt: "")

    entities = extractor._parse_response(
        "ENTITY: Alice | TYPE: person\nENTITY: Acme Inc | TYPE: organization\nENTITY: Thing | TYPE: unknown"
    )

    assert [entity.type for entity in entities] == [
        EntityType.PERSON,
        EntityType.ORGANIZATION,
        EntityType.OTHER,
    ]
