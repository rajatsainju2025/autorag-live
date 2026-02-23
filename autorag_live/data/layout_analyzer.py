import asyncio
import base64
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DocumentLayoutAnalyzer:
    """
    Document Layout Analyzer.

    This class uses a multimodal LLM (like GPT-4V or Claude 3 Vision) to analyze
    the layout of a document page (provided as an image) and extract structured
    content such as text blocks, tables, and figures. This is crucial for RAG
    systems dealing with complex PDFs where standard text extraction fails.
    """

    def __init__(self, vision_llm: Any, prompt_template: Optional[str] = None):
        """
        Initialize the DocumentLayoutAnalyzer.

        Args:
            vision_llm: A multimodal LLM client capable of processing images.
            prompt_template: The prompt to instruct the model on how to analyze the layout.
        """
        self.vision_llm = vision_llm
        self.prompt_template = prompt_template or (
            "Analyze the layout of this document page. Extract the content into structured JSON format. "
            "Identify text blocks, tables (as markdown or structured data), and figures (with captions). "
            "Return ONLY valid JSON with the following structure: "
            '{"text_blocks": ["..."], "tables": [{"caption": "...", "data": "..."}], "figures": [{"caption": "..."}]}'
        )

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to a base64 string.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    async def analyze_page(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single document page image.

        Args:
            image_path: The path to the image file (e.g., PNG or JPEG of a PDF page).

        Returns:
            A dictionary containing the structured layout analysis (text blocks, tables, figures).
        """
        base64_image = self._encode_image(image_path)

        # Construct the multimodal message payload
        # Note: The exact format depends on the specific vision_llm client being used.
        # This assumes a generic interface similar to LangChain's ChatMessage or OpenAI's API.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        try:
            # Assuming the vision_llm has an async generate or invoke method
            if hasattr(self.vision_llm, "agenerate"):
                response = await self.vision_llm.agenerate([messages])
                content = response.generations[0][0].text
            elif hasattr(self.vision_llm, "ainvoke"):
                response = await self.vision_llm.ainvoke(messages)
                content = response.content
            else:
                # Fallback to synchronous if async is not available
                loop = asyncio.get_running_loop()
                if hasattr(self.vision_llm, "invoke"):
                    response = await loop.run_in_executor(None, self.vision_llm.invoke, messages)
                    content = response.content
                else:
                    raise ValueError(
                        "Vision LLM must have 'agenerate', 'ainvoke', or 'invoke' method."
                    )

            # Parse the JSON response
            import json

            try:
                # Clean up potential markdown formatting around JSON
                clean_content = content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]

                structured_data = json.loads(clean_content)
                return structured_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from vision model response: {content}")
                return {"error": "Failed to parse JSON", "raw_response": content}

        except Exception as e:
            logger.error(f"Error during document layout analysis: {e}")
            raise
