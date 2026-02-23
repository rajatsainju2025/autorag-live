import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ContextualEmbedder:
    """
    Contextual Document Embeddings.

    This class implements a technique where the global context of a document
    is prepended to each of its chunks before generating embeddings. This helps
    the embedding model understand the broader context of a specific chunk,
    improving retrieval accuracy, especially for chunks that might be ambiguous
    on their own.
    """

    def __init__(
        self,
        embedding_model: Any,
        context_template: str = "Document Context: {context}\n\nChunk: {chunk}",
    ):
        """
        Initialize the ContextualEmbedder.

        Args:
            embedding_model: The model used to generate embeddings (must have an async `embed_documents` method).
            context_template: The template used to combine the context and the chunk.
        """
        self.embedding_model = embedding_model
        self.context_template = context_template

    async def embed_contextual_chunks(
        self, chunks: List[str], document_context: str
    ) -> List[List[float]]:
        """
        Embed chunks with prepended document context.

        Args:
            chunks: A list of text chunks from the document.
            document_context: The global context of the document (e.g., title, summary, or first paragraph).

        Returns:
            A list of embeddings corresponding to the contextualized chunks.
        """
        if not chunks:
            return []

        contextualized_chunks = [
            self.context_template.format(context=document_context, chunk=chunk) for chunk in chunks
        ]

        try:
            # Assuming the embedding model has an async embed_documents method
            if hasattr(self.embedding_model, "aembed_documents"):
                embeddings = await self.embedding_model.aembed_documents(contextualized_chunks)
            elif hasattr(self.embedding_model, "embed_documents"):
                # Fallback to synchronous if async is not available, running in a thread pool
                loop = asyncio.get_running_loop()
                embeddings = await loop.run_in_executor(
                    None, self.embedding_model.embed_documents, contextualized_chunks
                )
            else:
                raise ValueError(
                    "Embedding model must have 'aembed_documents' or 'embed_documents' method."
                )

            return embeddings
        except Exception as e:
            logger.error(f"Error generating contextual embeddings: {e}")
            raise

    async def process_document(
        self, document: Dict[str, Any], chunk_key: str = "chunks", context_key: str = "summary"
    ) -> Dict[str, Any]:
        """
        Process a full document, generating contextual embeddings for its chunks.

        Args:
            document: A dictionary representing the document.
            chunk_key: The key in the dictionary containing the list of chunks.
            context_key: The key in the dictionary containing the document context.

        Returns:
            The document dictionary updated with a new 'contextual_embeddings' key.
        """
        chunks = document.get(chunk_key, [])
        context = document.get(context_key, "")

        if not chunks:
            logger.warning("No chunks found in document.")
            document["contextual_embeddings"] = []
            return document

        if not context:
            logger.warning("No context found in document. Using empty context.")

        embeddings = await self.embed_contextual_chunks(chunks, context)
        document["contextual_embeddings"] = embeddings

        return document
