import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Knowledge Graph Entity Extractor.

    This class uses an LLM to extract entities and their relationships from text.
    This is a foundational step for building a Knowledge Graph (GraphRAG), allowing
    the system to understand the structured connections between concepts in the
    unstructured text.
    """

    def __init__(self, llm: Any, extraction_prompt: Optional[str] = None):
        """
        Initialize the EntityExtractor.

        Args:
            llm: The language model used for extraction.
            extraction_prompt: The prompt template instructing the LLM on how to extract entities.
        """
        self.llm = llm
        self.extraction_prompt = extraction_prompt or (
            "You are an expert knowledge graph builder. Extract entities and their relationships from the following text.\n"
            "Return ONLY a valid JSON object with two keys: 'entities' and 'relationships'.\n"
            "- 'entities' should be a list of objects with 'id' (unique name), 'type' (e.g., PERSON, ORGANIZATION, CONCEPT), and 'description'.\n"
            "- 'relationships' should be a list of objects with 'source' (entity id), 'target' (entity id), 'type' (e.g., WORKS_FOR, PART_OF), and 'description'.\n\n"
            "Text:\n{text}\n\n"
            "JSON Output:"
        )

    async def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from a given text.

        Args:
            text: The unstructured text to analyze.

        Returns:
            A dictionary containing the extracted 'entities' and 'relationships'.
        """
        if not text.strip():
            return {"entities": [], "relationships": []}

        prompt = self.extraction_prompt.format(text=text)

        try:
            # Assuming the LLM has an async generate or invoke method
            if hasattr(self.llm, "agenerate"):
                response = await self.llm.agenerate([prompt])
                content = response.generations[0][0].text
            elif hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(prompt)
                content = response.content
            else:
                # Fallback to synchronous if async is not available
                loop = asyncio.get_running_loop()
                if hasattr(self.llm, "invoke"):
                    response = await loop.run_in_executor(None, self.llm.invoke, prompt)
                    content = response.content
                else:
                    raise ValueError("LLM must have 'agenerate', 'ainvoke', or 'invoke' method.")

            # Parse the JSON response
            return self._parse_json_response(content)

        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            return {"entities": [], "relationships": [], "error": str(e)}

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Safely parse the JSON response from the LLM, handling potential markdown formatting.
        """
        clean_content = content.strip()

        # Remove markdown code blocks if present
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:]

        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]

        clean_content = clean_content.strip()

        try:
            data = json.loads(clean_content)
            # Ensure the expected keys exist
            if "entities" not in data:
                data["entities"] = []
            if "relationships" not in data:
                data["relationships"] = []
            return data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM response: {content}")
            return {"entities": [], "relationships": [], "error": "Failed to parse JSON"}

    async def process_documents(
        self, documents: List[Dict[str, Any]], text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Process a list of documents, extracting entities for each.

        Args:
            documents: A list of document dictionaries.
            text_key: The key containing the text to process.

        Returns:
            The documents with an added 'knowledge_graph' key containing the extracted data.
        """
        # Process concurrently for efficiency
        tasks = []
        for doc in documents:
            text = doc.get(text_key, "")
            tasks.append(self.extract(text))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, doc in enumerate(documents):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Error processing document {i}: {result}")
                doc["knowledge_graph"] = {"entities": [], "relationships": [], "error": str(result)}
            else:
                doc["knowledge_graph"] = result

        return documents
