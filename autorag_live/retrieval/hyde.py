"""
HyDE (Hypothetical Document Embeddings) Retrieval Implementation.

Based on: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2023)

HyDE improves retrieval by:
1. Generating hypothetical documents that would answer the query
2. Embedding the hypothetical documents instead of the raw query
3. Retrieving real documents similar to the hypothetical ones

This bridges the semantic gap between queries and documents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols and Types
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Embed text into vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval backends."""

    async def retrieve_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
    ) -> list["RetrievedDocument"]:
        """Retrieve documents by vector similarity."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


# ============================================================================
# Hypothetical Document Generation
# ============================================================================


class HyDEMode(Enum):
    """HyDE generation modes."""

    SINGLE = "single"  # Generate single hypothetical document
    MULTI = "multi"  # Generate multiple hypothetical documents
    DIVERSE = "diverse"  # Generate diverse hypothetical documents
    DOMAIN_SPECIFIC = "domain_specific"  # Domain-tailored generation


class DocumentType(Enum):
    """Types of hypothetical documents to generate."""

    PASSAGE = "passage"  # Web passage style
    ARTICLE = "article"  # Wikipedia article style
    QA_PAIR = "qa_pair"  # Question-answer pair
    CODE = "code"  # Code documentation
    SCIENTIFIC = "scientific"  # Scientific paper style
    CONVERSATIONAL = "conversational"  # Chat/dialogue style


@dataclass
class HyDEConfig:
    """Configuration for HyDE retrieval."""

    mode: HyDEMode = HyDEMode.SINGLE
    document_type: DocumentType = DocumentType.PASSAGE
    num_hypothetical: int = 1
    max_doc_length: int = 512
    temperature: float = 0.7
    aggregate_method: str = "mean"  # mean, max, weighted
    use_query_expansion: bool = False
    domain_context: str = ""


@dataclass
class HypotheticalDocument:
    """A generated hypothetical document."""

    content: str
    embedding: list[float] | None = None
    generation_prompt: str = ""
    confidence: float = 1.0


class HypotheticalDocumentGenerator(ABC):
    """Abstract base for hypothetical document generation."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        config: HyDEConfig,
    ) -> list[HypotheticalDocument]:
        """Generate hypothetical documents for query."""
        ...


class LLMHypotheticalGenerator(HypotheticalDocumentGenerator):
    """LLM-based hypothetical document generator."""

    PROMPT_TEMPLATES = {
        DocumentType.PASSAGE: """Write a web passage that directly answers this question.
The passage should be informative and contain specific details.

Question: {query}

Passage:""",
        DocumentType.ARTICLE: """Write a Wikipedia-style article section that answers this question.
Include factual information and cite-worthy content.

Question: {query}

Article Section:""",
        DocumentType.QA_PAIR: """Generate a detailed question-answer pair where the answer
addresses this query comprehensively.

Query: {query}

Q&A:""",
        DocumentType.CODE: """Write a code documentation snippet that explains how to solve
this programming question. Include code examples if relevant.

Question: {query}

Documentation:""",
        DocumentType.SCIENTIFIC: """Write a scientific abstract that addresses this research question.
Use technical language and cite-worthy findings.

Question: {query}

Abstract:""",
        DocumentType.CONVERSATIONAL: """Write a helpful assistant response that thoroughly
answers this user question with specific details.

User Question: {query}

Assistant Response:""",
    }

    DIVERSE_PERSPECTIVES = [
        "from a technical perspective",
        "from a beginner's viewpoint",
        "with historical context",
        "with practical examples",
        "focusing on key concepts",
    ]

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def generate(
        self,
        query: str,
        config: HyDEConfig,
    ) -> list[HypotheticalDocument]:
        """Generate hypothetical documents based on configuration."""
        if config.mode == HyDEMode.SINGLE:
            return await self._generate_single(query, config)
        elif config.mode == HyDEMode.MULTI:
            return await self._generate_multi(query, config)
        elif config.mode == HyDEMode.DIVERSE:
            return await self._generate_diverse(query, config)
        elif config.mode == HyDEMode.DOMAIN_SPECIFIC:
            return await self._generate_domain_specific(query, config)
        else:
            raise ValueError(f"Unknown HyDE mode: {config.mode}")

    async def _generate_single(
        self,
        query: str,
        config: HyDEConfig,
    ) -> list[HypotheticalDocument]:
        """Generate a single hypothetical document."""
        prompt = self._build_prompt(query, config.document_type)

        response = await self.llm.generate(
            prompt,
            max_tokens=config.max_doc_length,
            temperature=config.temperature,
        )

        return [
            HypotheticalDocument(
                content=response.strip(),
                generation_prompt=prompt,
            )
        ]

    async def _generate_multi(
        self,
        query: str,
        config: HyDEConfig,
    ) -> list[HypotheticalDocument]:
        """Generate multiple hypothetical documents."""
        prompt = self._build_prompt(query, config.document_type)
        docs = []

        for i in range(config.num_hypothetical):
            # Vary temperature slightly for diversity
            temp = config.temperature + (i * 0.1)
            temp = min(temp, 1.0)

            response = await self.llm.generate(
                prompt,
                max_tokens=config.max_doc_length,
                temperature=temp,
            )

            docs.append(
                HypotheticalDocument(
                    content=response.strip(),
                    generation_prompt=prompt,
                    confidence=1.0 - (i * 0.1),  # Decreasing confidence
                )
            )

        return docs

    async def _generate_diverse(
        self,
        query: str,
        config: HyDEConfig,
    ) -> list[HypotheticalDocument]:
        """Generate diverse hypothetical documents with different perspectives."""
        docs = []
        perspectives = self.DIVERSE_PERSPECTIVES[: config.num_hypothetical]

        for i, perspective in enumerate(perspectives):
            prompt = self._build_diverse_prompt(query, config.document_type, perspective)

            response = await self.llm.generate(
                prompt,
                max_tokens=config.max_doc_length,
                temperature=config.temperature,
            )

            docs.append(
                HypotheticalDocument(
                    content=response.strip(),
                    generation_prompt=prompt,
                    confidence=1.0,
                )
            )

        return docs

    async def _generate_domain_specific(
        self,
        query: str,
        config: HyDEConfig,
    ) -> list[HypotheticalDocument]:
        """Generate domain-specific hypothetical documents."""
        prompt = self._build_domain_prompt(query, config.document_type, config.domain_context)

        response = await self.llm.generate(
            prompt,
            max_tokens=config.max_doc_length,
            temperature=config.temperature,
        )

        return [
            HypotheticalDocument(
                content=response.strip(),
                generation_prompt=prompt,
            )
        ]

    def _build_prompt(self, query: str, doc_type: DocumentType) -> str:
        """Build generation prompt for document type."""
        template = self.PROMPT_TEMPLATES.get(doc_type, self.PROMPT_TEMPLATES[DocumentType.PASSAGE])
        return template.format(query=query)

    def _build_diverse_prompt(
        self,
        query: str,
        doc_type: DocumentType,
        perspective: str,
    ) -> str:
        """Build prompt with specific perspective."""
        base_template = self.PROMPT_TEMPLATES.get(
            doc_type, self.PROMPT_TEMPLATES[DocumentType.PASSAGE]
        )
        modified = base_template.replace(
            "Question:",
            f"Question ({perspective}):",
        )
        return modified.format(query=query)

    def _build_domain_prompt(
        self,
        query: str,
        doc_type: DocumentType,
        domain_context: str,
    ) -> str:
        """Build domain-specific prompt."""
        base_template = self.PROMPT_TEMPLATES.get(
            doc_type, self.PROMPT_TEMPLATES[DocumentType.PASSAGE]
        )
        domain_prefix = f"Domain Context: {domain_context}\n\n"
        return domain_prefix + base_template.format(query=query)


# ============================================================================
# Embedding Aggregation
# ============================================================================


class EmbeddingAggregator:
    """Aggregate multiple embeddings into one."""

    @staticmethod
    def aggregate(
        embeddings: list[list[float]],
        method: str = "mean",
        weights: list[float] | None = None,
    ) -> list[float]:
        """Aggregate embeddings using specified method."""
        if not embeddings:
            raise ValueError("No embeddings to aggregate")

        if len(embeddings) == 1:
            return embeddings[0]

        if method == "mean":
            return EmbeddingAggregator._mean_aggregate(embeddings)
        elif method == "max":
            return EmbeddingAggregator._max_aggregate(embeddings)
        elif method == "weighted":
            return EmbeddingAggregator._weighted_aggregate(embeddings, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    @staticmethod
    def _mean_aggregate(embeddings: list[list[float]]) -> list[float]:
        """Mean pooling of embeddings."""
        dim = len(embeddings[0])
        result = [0.0] * dim

        for emb in embeddings:
            for i in range(dim):
                result[i] += emb[i]

        n = len(embeddings)
        return [v / n for v in result]

    @staticmethod
    def _max_aggregate(embeddings: list[list[float]]) -> list[float]:
        """Max pooling of embeddings."""
        dim = len(embeddings[0])
        result = [float("-inf")] * dim

        for emb in embeddings:
            for i in range(dim):
                result[i] = max(result[i], emb[i])

        return result

    @staticmethod
    def _weighted_aggregate(
        embeddings: list[list[float]],
        weights: list[float] | None,
    ) -> list[float]:
        """Weighted average of embeddings."""
        if weights is None:
            # Use uniform weights
            weights = [1.0 / len(embeddings)] * len(embeddings)

        if len(weights) != len(embeddings):
            raise ValueError("Weights must match number of embeddings")

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        dim = len(embeddings[0])
        result = [0.0] * dim

        for emb, weight in zip(embeddings, weights):
            for i in range(dim):
                result[i] += emb[i] * weight

        return result


# ============================================================================
# HyDE Retriever
# ============================================================================


@dataclass
class HyDEResult:
    """Result from HyDE retrieval."""

    documents: list[RetrievedDocument]
    hypothetical_docs: list[HypotheticalDocument]
    query_embedding: list[float] | None = None
    aggregated_embedding: list[float] | None = None
    retrieval_method: str = "hyde"


class HyDERetriever:
    """
    HyDE (Hypothetical Document Embeddings) Retriever.

    Improves dense retrieval by generating hypothetical documents
    and using their embeddings for retrieval instead of raw queries.
    """

    def __init__(
        self,
        generator: HypotheticalDocumentGenerator,
        embedder: EmbeddingProtocol,
        retriever: RetrieverProtocol,
        config: HyDEConfig | None = None,
    ):
        """
        Initialize HyDE retriever.

        Args:
            generator: Hypothetical document generator
            embedder: Embedding provider
            retriever: Backend retriever
            config: HyDE configuration
        """
        self.generator = generator
        self.embedder = embedder
        self.retriever = retriever
        self.config = config or HyDEConfig()
        self.aggregator = EmbeddingAggregator()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        config: HyDEConfig | None = None,
    ) -> HyDEResult:
        """
        Retrieve documents using HyDE.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            config: Optional config override

        Returns:
            HyDEResult with retrieved documents
        """
        cfg = config or self.config

        # Step 1: Generate hypothetical documents
        hypothetical_docs = await self.generator.generate(query, cfg)

        # Step 2: Embed hypothetical documents
        doc_texts = [doc.content for doc in hypothetical_docs]
        embeddings = await self.embedder.embed_batch(doc_texts)

        # Store embeddings in hypothetical docs
        for doc, emb in zip(hypothetical_docs, embeddings):
            doc.embedding = emb

        # Step 3: Aggregate embeddings
        weights = None
        if cfg.aggregate_method == "weighted":
            weights = [doc.confidence for doc in hypothetical_docs]

        aggregated = self.aggregator.aggregate(
            embeddings,
            method=cfg.aggregate_method,
            weights=weights,
        )

        # Step 4: Retrieve using aggregated embedding
        documents = await self.retriever.retrieve_by_vector(
            vector=aggregated,
            top_k=top_k,
        )

        return HyDEResult(
            documents=documents,
            hypothetical_docs=hypothetical_docs,
            aggregated_embedding=aggregated,
            retrieval_method="hyde",
        )

    async def retrieve_with_fallback(
        self,
        query: str,
        top_k: int = 5,
        config: HyDEConfig | None = None,
    ) -> HyDEResult:
        """
        Retrieve with fallback to standard retrieval if HyDE fails.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            config: Optional config override

        Returns:
            HyDEResult with retrieved documents
        """
        try:
            result = await self.retrieve(query, top_k, config)

            # Check if we got meaningful results
            if result.documents:
                return result

            # Fall back to standard retrieval
            return await self._standard_retrieve(query, top_k)

        except Exception:
            # Fall back to standard retrieval on any error
            return await self._standard_retrieve(query, top_k)

    async def _standard_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> HyDEResult:
        """Standard retrieval without HyDE."""
        query_embedding = await self.embedder.embed(query)
        documents = await self.retriever.retrieve_by_vector(
            vector=query_embedding,
            top_k=top_k,
        )

        return HyDEResult(
            documents=documents,
            hypothetical_docs=[],
            query_embedding=query_embedding,
            retrieval_method="standard",
        )


# ============================================================================
# Multi-Document HyDE (for complex queries)
# ============================================================================


@dataclass
class MultiDocHyDEConfig(HyDEConfig):
    """Configuration for multi-document HyDE."""

    num_document_types: int = 3
    combine_retrievals: bool = True
    max_total_docs: int = 10


class MultiDocHyDERetriever:
    """
    Multi-Document HyDE Retriever.

    Generates hypothetical documents of multiple types and
    combines retrievals for better coverage.
    """

    DEFAULT_DOC_TYPES = [
        DocumentType.PASSAGE,
        DocumentType.ARTICLE,
        DocumentType.QA_PAIR,
    ]

    def __init__(
        self,
        generator: HypotheticalDocumentGenerator,
        embedder: EmbeddingProtocol,
        retriever: RetrieverProtocol,
        config: MultiDocHyDEConfig | None = None,
    ):
        """Initialize multi-document HyDE retriever."""
        self.generator = generator
        self.embedder = embedder
        self.retriever = retriever
        self.config = config or MultiDocHyDEConfig()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_types: list[DocumentType] | None = None,
    ) -> HyDEResult:
        """
        Retrieve using multiple document types.

        Args:
            query: User query
            top_k: Documents per type
            document_types: Types to generate

        Returns:
            Combined HyDEResult
        """
        doc_types = document_types or self.DEFAULT_DOC_TYPES[: self.config.num_document_types]

        all_hypothetical: list[HypotheticalDocument] = []
        all_embeddings: list[list[float]] = []

        # Generate hypothetical docs for each type
        for doc_type in doc_types:
            type_config = HyDEConfig(
                mode=self.config.mode,
                document_type=doc_type,
                num_hypothetical=1,
                max_doc_length=self.config.max_doc_length,
                temperature=self.config.temperature,
            )

            hypothetical = await self.generator.generate(query, type_config)
            all_hypothetical.extend(hypothetical)

            # Embed
            for doc in hypothetical:
                emb = await self.embedder.embed(doc.content)
                doc.embedding = emb
                all_embeddings.append(emb)

        if self.config.combine_retrievals:
            # Aggregate all embeddings
            aggregated = EmbeddingAggregator.aggregate(
                all_embeddings,
                method=self.config.aggregate_method,
            )

            documents = await self.retriever.retrieve_by_vector(
                vector=aggregated,
                top_k=self.config.max_total_docs,
            )
        else:
            # Retrieve separately and merge
            all_docs: list[RetrievedDocument] = []
            per_type_k = top_k

            for emb in all_embeddings:
                docs = await self.retriever.retrieve_by_vector(
                    vector=emb,
                    top_k=per_type_k,
                )
                all_docs.extend(docs)

            # Deduplicate and sort
            documents = self._deduplicate_and_rank(
                all_docs,
                self.config.max_total_docs,
            )
            aggregated = None

        return HyDEResult(
            documents=documents,
            hypothetical_docs=all_hypothetical,
            aggregated_embedding=aggregated if self.config.combine_retrievals else None,
            retrieval_method="multi_doc_hyde",
        )

    def _deduplicate_and_rank(
        self,
        documents: list[RetrievedDocument],
        max_docs: int,
    ) -> list[RetrievedDocument]:
        """Deduplicate documents and rank by score."""
        seen_ids: set[str] = set()
        unique_docs: list[RetrievedDocument] = []

        for doc in documents:
            doc_id = doc.doc_id or hash(doc.content)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        # Sort by score descending
        unique_docs.sort(key=lambda d: d.score, reverse=True)

        return unique_docs[:max_docs]


# ============================================================================
# Query-Aware HyDE
# ============================================================================


class QueryAwareHyDE:
    """
    Query-Aware HyDE that adapts generation based on query analysis.

    Automatically selects document type and generation parameters
    based on query characteristics.
    """

    QUERY_PATTERNS = {
        "how to": DocumentType.PASSAGE,
        "what is": DocumentType.ARTICLE,
        "why": DocumentType.ARTICLE,
        "code": DocumentType.CODE,
        "implement": DocumentType.CODE,
        "research": DocumentType.SCIENTIFIC,
        "study": DocumentType.SCIENTIFIC,
        "help": DocumentType.CONVERSATIONAL,
    }

    def __init__(
        self,
        generator: HypotheticalDocumentGenerator,
        embedder: EmbeddingProtocol,
        retriever: RetrieverProtocol,
    ):
        """Initialize query-aware HyDE."""
        self.generator = generator
        self.embedder = embedder
        self.retriever = retriever

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> HyDEResult:
        """
        Retrieve with automatic query analysis.

        Args:
            query: User query
            top_k: Number of documents

        Returns:
            HyDEResult with retrieved documents
        """
        # Analyze query to determine best document type
        doc_type = self._analyze_query(query)

        # Determine optimal config
        config = self._build_config(query, doc_type)

        # Create HyDE retriever and retrieve
        hyde_retriever = HyDERetriever(
            generator=self.generator,
            embedder=self.embedder,
            retriever=self.retriever,
            config=config,
        )

        return await hyde_retriever.retrieve(query, top_k, config)

    def _analyze_query(self, query: str) -> DocumentType:
        """Analyze query to determine best document type."""
        query_lower = query.lower()

        for pattern, doc_type in self.QUERY_PATTERNS.items():
            if pattern in query_lower:
                return doc_type

        # Default to passage
        return DocumentType.PASSAGE

    def _build_config(self, query: str, doc_type: DocumentType) -> HyDEConfig:
        """Build optimal config for query."""
        # Complex queries benefit from multiple hypotheticals
        is_complex = len(query.split()) > 10 or "and" in query.lower()

        return HyDEConfig(
            mode=HyDEMode.MULTI if is_complex else HyDEMode.SINGLE,
            document_type=doc_type,
            num_hypothetical=3 if is_complex else 1,
            max_doc_length=512,
            temperature=0.7,
            aggregate_method="mean",
        )


# ============================================================================
# HyDE with Query Expansion
# ============================================================================


class HyDEWithExpansion:
    """
    HyDE combined with query expansion for improved recall.

    Generates hypothetical documents and also expands the query
    to capture more relevant documents.
    """

    def __init__(
        self,
        generator: HypotheticalDocumentGenerator,
        embedder: EmbeddingProtocol,
        retriever: RetrieverProtocol,
        llm: LLMProtocol,
    ):
        """Initialize HyDE with expansion."""
        self.generator = generator
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expansion_count: int = 3,
    ) -> HyDEResult:
        """
        Retrieve with HyDE and query expansion.

        Args:
            query: User query
            top_k: Number of documents
            expansion_count: Number of query expansions

        Returns:
            Combined HyDEResult
        """
        # Generate expanded queries
        expanded_queries = await self._expand_query(query, expansion_count)

        all_hypothetical: list[HypotheticalDocument] = []
        all_embeddings: list[list[float]] = []

        # Generate hypothetical docs for original and expanded queries
        all_queries = [query] + expanded_queries

        for q in all_queries:
            config = HyDEConfig(
                mode=HyDEMode.SINGLE,
                document_type=DocumentType.PASSAGE,
            )

            hypothetical = await self.generator.generate(q, config)
            all_hypothetical.extend(hypothetical)

            for doc in hypothetical:
                emb = await self.embedder.embed(doc.content)
                doc.embedding = emb
                all_embeddings.append(emb)

        # Aggregate all embeddings
        aggregated = EmbeddingAggregator.aggregate(
            all_embeddings,
            method="mean",
        )

        # Retrieve
        documents = await self.retriever.retrieve_by_vector(
            vector=aggregated,
            top_k=top_k,
        )

        return HyDEResult(
            documents=documents,
            hypothetical_docs=all_hypothetical,
            aggregated_embedding=aggregated,
            retrieval_method="hyde_with_expansion",
        )

    async def _expand_query(
        self,
        query: str,
        count: int,
    ) -> list[str]:
        """Expand query into related queries."""
        prompt = f"""Generate {count} alternative ways to ask this question.
Each alternative should capture the same intent but use different words.

Original Question: {query}

Alternative Questions:"""

        response = await self.llm.generate(prompt, max_tokens=200)

        # Parse response into individual queries
        lines = response.strip().split("\n")
        expansions = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("Alternative"):
                # Remove numbering if present
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                    line = line.split(")", 1)[-1].strip()
                if line:
                    expansions.append(line)

        return expansions[:count]


# ============================================================================
# Factory Functions
# ============================================================================


def create_hyde_retriever(
    llm: LLMProtocol,
    embedder: EmbeddingProtocol,
    retriever: RetrieverProtocol,
    mode: HyDEMode = HyDEMode.SINGLE,
    document_type: DocumentType = DocumentType.PASSAGE,
) -> HyDERetriever:
    """
    Create a configured HyDE retriever.

    Args:
        llm: LLM for hypothetical document generation
        embedder: Embedding provider
        retriever: Backend retriever
        mode: HyDE generation mode
        document_type: Type of hypothetical documents

    Returns:
        Configured HyDERetriever
    """
    generator = LLMHypotheticalGenerator(llm)
    config = HyDEConfig(
        mode=mode,
        document_type=document_type,
    )

    return HyDERetriever(
        generator=generator,
        embedder=embedder,
        retriever=retriever,
        config=config,
    )


def create_query_aware_hyde(
    llm: LLMProtocol,
    embedder: EmbeddingProtocol,
    retriever: RetrieverProtocol,
) -> QueryAwareHyDE:
    """
    Create a query-aware HyDE retriever.

    Args:
        llm: LLM for hypothetical document generation
        embedder: Embedding provider
        retriever: Backend retriever

    Returns:
        QueryAwareHyDE instance
    """
    generator = LLMHypotheticalGenerator(llm)

    return QueryAwareHyDE(
        generator=generator,
        embedder=embedder,
        retriever=retriever,
    )


def create_multi_doc_hyde(
    llm: LLMProtocol,
    embedder: EmbeddingProtocol,
    retriever: RetrieverProtocol,
    num_types: int = 3,
) -> MultiDocHyDERetriever:
    """
    Create a multi-document HyDE retriever.

    Args:
        llm: LLM for hypothetical document generation
        embedder: Embedding provider
        retriever: Backend retriever
        num_types: Number of document types to generate

    Returns:
        MultiDocHyDERetriever instance
    """
    generator = LLMHypotheticalGenerator(llm)
    config = MultiDocHyDEConfig(
        num_document_types=num_types,
        combine_retrievals=True,
    )

    return MultiDocHyDERetriever(
        generator=generator,
        embedder=embedder,
        retriever=retriever,
        config=config,
    )


# ============================================================================
# Evaluation Utilities
# ============================================================================


@dataclass
class HyDEEvaluation:
    """Evaluation metrics for HyDE retrieval."""

    hyde_recall: float
    standard_recall: float
    hyde_mrr: float
    standard_mrr: float
    improvement: float
    query: str
    num_hypothetical: int


class HyDEEvaluator:
    """Evaluator for comparing HyDE vs standard retrieval."""

    def __init__(
        self,
        hyde_retriever: HyDERetriever,
        embedder: EmbeddingProtocol,
        retriever: RetrieverProtocol,
    ):
        """Initialize evaluator."""
        self.hyde_retriever = hyde_retriever
        self.embedder = embedder
        self.retriever = retriever

    async def evaluate(
        self,
        query: str,
        relevant_doc_ids: set[str],
        top_k: int = 10,
    ) -> HyDEEvaluation:
        """
        Evaluate HyDE vs standard retrieval.

        Args:
            query: Test query
            relevant_doc_ids: Ground truth relevant document IDs
            top_k: Number of documents to retrieve

        Returns:
            HyDEEvaluation with metrics
        """
        # HyDE retrieval
        hyde_result = await self.hyde_retriever.retrieve(query, top_k)
        hyde_doc_ids = [d.doc_id for d in hyde_result.documents]

        # Standard retrieval
        query_emb = await self.embedder.embed(query)
        standard_docs = await self.retriever.retrieve_by_vector(query_emb, top_k)
        standard_doc_ids = [d.doc_id for d in standard_docs]

        # Calculate metrics
        hyde_recall = self._recall(hyde_doc_ids, relevant_doc_ids)
        standard_recall = self._recall(standard_doc_ids, relevant_doc_ids)
        hyde_mrr = self._mrr(hyde_doc_ids, relevant_doc_ids)
        standard_mrr = self._mrr(standard_doc_ids, relevant_doc_ids)

        improvement = hyde_recall - standard_recall

        return HyDEEvaluation(
            hyde_recall=hyde_recall,
            standard_recall=standard_recall,
            hyde_mrr=hyde_mrr,
            standard_mrr=standard_mrr,
            improvement=improvement,
            query=query,
            num_hypothetical=len(hyde_result.hypothetical_docs),
        )

    def _recall(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:
        """Calculate recall@k."""
        if not relevant_ids:
            return 0.0

        found = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
        return found / len(relevant_ids)

    def _mrr(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating HyDE retrieval."""
    # This would use actual implementations
    # llm = OpenAILLM(api_key="...")
    # embedder = OpenAIEmbeddings(api_key="...")
    # retriever = VectorStoreRetriever(...)

    # Basic HyDE retrieval
    # hyde = create_hyde_retriever(llm, embedder, retriever)
    # result = await hyde.retrieve("What is machine learning?")

    # Query-aware HyDE
    # qa_hyde = create_query_aware_hyde(llm, embedder, retriever)
    # result = await qa_hyde.retrieve("How to implement a neural network in Python?")

    # Multi-document HyDE
    # md_hyde = create_multi_doc_hyde(llm, embedder, retriever)
    # result = await md_hyde.retrieve("Explain transformer architecture")

    print("HyDE Retrieval Implementation Ready")
    print("=" * 50)
    print("Features:")
    print("- Single/Multi/Diverse hypothetical document generation")
    print("- Multiple document types (passage, article, QA, code, scientific)")
    print("- Query-aware automatic configuration")
    print("- Embedding aggregation (mean, max, weighted)")
    print("- Query expansion integration")
    print("- Evaluation utilities")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
