import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StepBackPromptingAgent:
    """
    Step-Back Prompting Agent.

    This agent implements the Step-Back Prompting technique. It takes a specific,
    detailed query and uses an LLM to generate a more abstract, higher-level
    question (the "step-back" question). It then retrieves context for both the
    original and the step-back question, providing a richer, more comprehensive
    context for the final answer generation.
    """

    def __init__(self, llm: Any, retriever: Any, step_back_prompt: Optional[str] = None):
        """
        Initialize the StepBackPromptingAgent.

        Args:
            llm: The language model used to generate the step-back question.
            retriever: The retrieval component (must have an async `retrieve` or `get_relevant_documents` method).
            step_back_prompt: The prompt template instructing the LLM to generate the step-back question.
        """
        self.llm = llm
        self.retriever = retriever
        self.step_back_prompt = step_back_prompt or (
            "You are an expert at understanding the core concepts behind specific questions. "
            "Given the following specific question, generate a more abstract, higher-level "
            "question that captures the underlying principles or broader context needed to answer it.\n\n"
            "Specific Question: {query}\n\n"
            "Step-Back Question:"
        )

    async def generate_step_back_question(self, query: str) -> str:
        """
        Generate the abstract step-back question.
        """
        prompt = self.step_back_prompt.format(query=query)

        try:
            if hasattr(self.llm, "agenerate"):
                response = await self.llm.agenerate([prompt])
                return response.generations[0][0].text.strip()
            elif hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(prompt)
                return response.content.strip()
            else:
                loop = asyncio.get_running_loop()
                if hasattr(self.llm, "invoke"):
                    response = await loop.run_in_executor(None, self.llm.invoke, prompt)
                    return response.content.strip()
                else:
                    raise ValueError("LLM must have 'agenerate', 'ainvoke', or 'invoke' method.")
        except Exception as e:
            logger.error(f"Error generating step-back question: {e}")
            # Fallback to the original query if generation fails
            return query

    async def _retrieve_docs(self, query: str) -> List[Dict[str, Any]]:
        """
        Helper to retrieve documents using the provided retriever.
        """
        try:
            if hasattr(self.retriever, "aretrieve"):
                return await self.retriever.aretrieve(query)
            elif hasattr(self.retriever, "aget_relevant_documents"):
                return await self.retriever.aget_relevant_documents(query)
            else:
                loop = asyncio.get_running_loop()
                if hasattr(self.retriever, "retrieve"):
                    return await loop.run_in_executor(None, self.retriever.retrieve, query)
                elif hasattr(self.retriever, "get_relevant_documents"):
                    return await loop.run_in_executor(
                        None, self.retriever.get_relevant_documents, query
                    )
                else:
                    raise ValueError("Retriever must have a valid retrieval method.")
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query}': {e}")
            return []

    async def process(self, query: str) -> Dict[str, Any]:
        """
        Execute the full step-back prompting workflow.

        Args:
            query: The original user query.

        Returns:
            A dictionary containing the original query, the step-back question,
            and the combined retrieved context.
        """
        # 1. Generate the step-back question
        step_back_question = await self.generate_step_back_question(query)
        logger.info(f"Original Query: {query}")
        logger.info(f"Step-Back Question: {step_back_question}")

        # 2. Retrieve documents for both queries concurrently
        original_docs_task = self._retrieve_docs(query)
        step_back_docs_task = self._retrieve_docs(step_back_question)

        original_docs, step_back_docs = await asyncio.gather(
            original_docs_task, step_back_docs_task
        )

        # 3. Combine and deduplicate documents (assuming docs have an 'id' or 'content' field)
        combined_docs = []
        seen_content = set()

        for doc in original_docs + step_back_docs:
            # Use 'content' or 'text' as the deduplication key if 'id' is not present
            content_key = doc.get("id", doc.get("content", doc.get("text", str(doc))))
            if content_key not in seen_content:
                seen_content.add(content_key)
                combined_docs.append(doc)

        return {
            "original_query": query,
            "step_back_question": step_back_question,
            "original_context": original_docs,
            "step_back_context": step_back_docs,
            "combined_context": combined_docs,
        }
