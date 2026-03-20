import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from autorag_live.core.agent_policy import AgentPolicy
from autorag_live.core.state_manager import StateManager

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Knowledge Graph Entity Extractor.

    This class uses an LLM to extract entities and their relationships from text.
    This is a foundational step for building a Knowledge Graph (GraphRAG), allowing
    the system to understand the structured connections between concepts in the
    unstructured text.
    """

    def __init__(
        self,
        llm: Any,
        extraction_prompt: Optional[str] = None,
        state_manager: Optional[StateManager] = None,
        policy: Optional[AgentPolicy] = None,
        max_concurrency: int = 8,
    ):
        """
        Initialize the EntityExtractor.

        Args:
            llm: The language model used for extraction.
            extraction_prompt: The prompt template instructing the LLM on how to extract entities.
        """
        self.llm = llm
        self.state_manager = state_manager
        self.policy = policy
        self.max_concurrency = max(1, max_concurrency)
        self._llm_runner = self._resolve_llm_runner(llm)

        self.extraction_prompt = extraction_prompt or (
            "You are an expert knowledge graph builder. Extract entities and their relationships from the following text.\n"
            "Return ONLY a valid JSON object with two keys: 'entities' and 'relationships'.\n"
            "- 'entities' should be a list of objects with 'id' (unique name), 'type' (e.g., PERSON, ORGANIZATION, CONCEPT), and 'description'.\n"
            "- 'relationships' should be a list of objects with 'source' (entity id), 'target' (entity id), 'type' (e.g., WORKS_FOR, PART_OF), and 'description'.\n\n"
            "Text:\n{text}\n\n"
            "JSON Output:"
        )

    def _resolve_llm_runner(self, llm: Any):
        """Resolve the fastest supported LLM call path once during initialization."""
        if hasattr(llm, "agenerate"):
            return self._run_agenerate
        if hasattr(llm, "ainvoke"):
            return self._run_ainvoke
        if hasattr(llm, "invoke"):
            return self._run_invoke
        return None

    async def _run_agenerate(self, prompt: str) -> str:
        response = await self.llm.agenerate([prompt])
        return response.generations[0][0].text

    async def _run_ainvoke(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def _run_invoke(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self.llm.invoke, prompt)
        return response.content

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
            if self._llm_runner is None:
                raise ValueError("LLM must have 'agenerate', 'ainvoke', or 'invoke' method.")

            content = await self._llm_runner(prompt)

            # Parse the JSON response
            parsed = self._parse_json_response(content)

            # Allow a policy to inspect/decide based on the parsed result and current state
            try:
                policy = self.policy
                if policy is not None:
                    state_manager = self.state_manager
                    state_snapshot = state_manager.snapshot() if state_manager is not None else {}
                    # Policy may return instructions or modifications; for now we log the decision
                    decision = policy.decide(
                        {"text": text, "response": parsed}, state=state_snapshot
                    )
                    logger.debug(f"Policy decision: {decision}")
            except Exception:
                logger.exception("Policy decision failed")

            return parsed

        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            return {"entities": [], "relationships": [], "error": str(e)}

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Safely parse the JSON response from the LLM, handling potential markdown formatting.
        """
        clean_content = content.strip()
        if clean_content.startswith("```"):
            first_newline = clean_content.find("\n")
            if first_newline != -1:
                clean_content = clean_content[first_newline + 1 :]
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
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def extract_with_limit(document: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.extract(document.get(text_key, ""))

        results = await asyncio.gather(
            *(extract_with_limit(doc) for doc in documents),
            return_exceptions=True,
        )

        for index, (doc, result) in enumerate(zip(documents, results)):
            if isinstance(result, Exception):
                logger.error(f"Error processing document {index}: {result}")
                doc["knowledge_graph"] = {"entities": [], "relationships": [], "error": str(result)}
            else:
                doc["knowledge_graph"] = result

        return documents
