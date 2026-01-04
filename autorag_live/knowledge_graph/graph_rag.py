"""
Graph-RAG Module for Entity-Aware Retrieval.

Implements knowledge graph-enhanced retrieval that combines
entity extraction, graph traversal, and vector similarity.

Key Features:
1. Entity extraction from queries and documents
2. Knowledge graph construction and management
3. Graph-based context expansion
4. Entity-aware retrieval ranking
5. Subgraph extraction for context

References:
- Graph-RAG: Microsoft Research, 2024
- KGRAG: Knowledge Graph Enhanced RAG

Example:
    >>> graph_rag = GraphRAG(entity_linker, graph, retriever)
    >>> result = await graph_rag.retrieve("Who founded OpenAI?", k=5)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Interfaces
# =============================================================================


class EntityExtractorProtocol(Protocol):
    """Protocol for entity extraction."""

    async def extract(self, text: str) -> List["Entity"]:
        """Extract entities from text."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever interface."""

    async def retrieve(self, query: str, k: int = 5) -> List["Document"]:
        """Retrieve relevant documents."""
        ...


class EmbedderProtocol(Protocol):
    """Protocol for embedder interface."""

    async def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class EntityType(str, Enum):
    """Types of entities."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    EVENT = "event"
    PRODUCT = "product"
    CONCEPT = "concept"
    OTHER = "other"


class RelationType(str, Enum):
    """Types of relationships."""

    IS_A = "is_a"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    FOUNDED = "founded"
    CREATED = "created"
    RELATED_TO = "related_to"
    HAS_PROPERTY = "has_property"


@dataclass
class Entity:
    """
    An entity in the knowledge graph.

    Attributes:
        id: Unique identifier
        name: Entity name
        type: Entity type
        aliases: Alternative names
        properties: Entity properties
        embedding: Vector embedding
    """

    id: str
    name: str
    type: EntityType = EntityType.OTHER
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source_docs: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass
class Relation:
    """
    A relation between entities.

    Attributes:
        source: Source entity ID
        target: Target entity ID
        type: Relation type
        properties: Relation properties
        confidence: Confidence score
    """

    source: str
    target: str
    type: RelationType = RelationType.RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_doc: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.type))


@dataclass
class Document:
    """A document for retrieval."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    entities: List[Entity] = field(default_factory=list)


@dataclass
class Subgraph:
    """
    A subgraph extracted from the knowledge graph.

    Attributes:
        entities: Entities in subgraph
        relations: Relations in subgraph
        central_entity: The central entity
        depth: Traversal depth
    """

    entities: List[Entity]
    relations: List[Relation]
    central_entity: Optional[Entity] = None
    depth: int = 1

    def to_context_string(self) -> str:
        """Convert subgraph to context string."""
        lines = []

        # Add entity descriptions
        for entity in self.entities:
            props = ", ".join(f"{k}: {v}" for k, v in entity.properties.items())
            lines.append(f"- {entity.name} ({entity.type.value}): {props}")

        # Add relations
        entity_map = {e.id: e.name for e in self.entities}
        for rel in self.relations:
            source_name = entity_map.get(rel.source, rel.source)
            target_name = entity_map.get(rel.target, rel.target)
            lines.append(f"- {source_name} --[{rel.type.value}]--> {target_name}")

        return "\n".join(lines)


@dataclass
class GraphRAGResult:
    """
    Result from Graph-RAG retrieval.

    Attributes:
        documents: Retrieved documents
        entities: Extracted entities
        subgraph: Knowledge subgraph
        query_entities: Entities from query
        combined_score: Combined relevance score
    """

    documents: List[Document]
    entities: List[Entity]
    subgraph: Optional[Subgraph] = None
    query_entities: List[Entity] = field(default_factory=list)
    combined_score: float = 0.0
    context_string: str = ""


# =============================================================================
# Knowledge Graph
# =============================================================================


class KnowledgeGraph:
    """
    In-memory knowledge graph.

    Stores entities and relations with efficient lookups.
    """

    def __init__(self):
        """Initialize graph."""
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._outgoing: Dict[str, List[Relation]] = defaultdict(list)
        self._incoming: Dict[str, List[Relation]] = defaultdict(list)
        self._name_index: Dict[str, Set[str]] = defaultdict(set)

    def add_entity(self, entity: Entity) -> None:
        """Add entity to graph."""
        self._entities[entity.id] = entity
        self._name_index[entity.name.lower()].add(entity.id)
        for alias in entity.aliases:
            self._name_index[alias.lower()].add(entity.id)

    def add_relation(self, relation: Relation) -> None:
        """Add relation to graph."""
        self._relations.append(relation)
        self._outgoing[relation.source].append(relation)
        self._incoming[relation.target].append(relation)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> List[Entity]:
        """Get entities by name."""
        ids = self._name_index.get(name.lower(), set())
        return [self._entities[id] for id in ids if id in self._entities]

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> List[Tuple[Entity, Relation]]:
        """
        Get neighboring entities.

        Args:
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (entity, relation) tuples
        """
        neighbors = []

        if direction in ("outgoing", "both"):
            for rel in self._outgoing[entity_id]:
                if rel.target in self._entities:
                    neighbors.append((self._entities[rel.target], rel))

        if direction in ("incoming", "both"):
            for rel in self._incoming[entity_id]:
                if rel.source in self._entities:
                    neighbors.append((self._entities[rel.source], rel))

        return neighbors

    def get_subgraph(
        self,
        entity_id: str,
        depth: int = 2,
        max_entities: int = 50,
    ) -> Subgraph:
        """
        Extract subgraph around entity.

        Args:
            entity_id: Central entity ID
            depth: Traversal depth
            max_entities: Maximum entities to include

        Returns:
            Extracted subgraph
        """
        central = self._entities.get(entity_id)
        if not central:
            return Subgraph(entities=[], relations=[])

        visited: Set[str] = {entity_id}
        entities: List[Entity] = [central]
        relations: List[Relation] = []
        frontier: Set[str] = {entity_id}

        for _ in range(depth):
            if len(entities) >= max_entities:
                break

            new_frontier: Set[str] = set()

            for eid in frontier:
                for neighbor, rel in self.get_neighbors(eid):
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        entities.append(neighbor)
                        new_frontier.add(neighbor.id)

                    relations.append(rel)

                    if len(entities) >= max_entities:
                        break

            frontier = new_frontier

        # Deduplicate relations
        unique_relations = list({r: r for r in relations}.values())

        return Subgraph(
            entities=entities,
            relations=unique_relations,
            central_entity=central,
            depth=depth,
        )

    def search_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Search entities by name.

        Args:
            query: Search query
            entity_type: Filter by type
            limit: Maximum results

        Returns:
            Matching entities
        """
        query_lower = query.lower()
        results = []

        for entity in self._entities.values():
            if entity_type and entity.type != entity_type:
                continue

            # Check name and aliases
            if query_lower in entity.name.lower():
                results.append(entity)
            elif any(query_lower in alias.lower() for alias in entity.aliases):
                results.append(entity)

            if len(results) >= limit:
                break

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "entity_types": len(set(e.type for e in self._entities.values())),
            "relation_types": len(set(r.type for r in self._relations)),
        }


# =============================================================================
# Entity Extraction
# =============================================================================


class EntityExtractor(ABC):
    """Abstract base class for entity extraction."""

    @abstractmethod
    async def extract(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        pass


class RuleBasedEntityExtractor(EntityExtractor):
    """
    Rule-based entity extractor.

    Uses patterns and heuristics for entity extraction.
    """

    def __init__(self):
        """Initialize extractor."""
        import re

        # Simple patterns for common entity types
        self.patterns = {
            EntityType.DATE: re.compile(
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
                re.IGNORECASE,
            ),
            EntityType.ORGANIZATION: re.compile(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Co)\.?)\b"
            ),
        }

    async def extract(self, text: str) -> List[Entity]:
        """Extract entities using rules."""
        entities = []
        entity_id = 0

        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append(
                    Entity(
                        id=f"rule_{entity_id}",
                        name=match.group(),
                        type=entity_type,
                    )
                )
                entity_id += 1

        # Extract capitalized phrases (likely proper nouns)
        import re

        caps_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
        for match in caps_pattern.finditer(text):
            name = match.group()
            if not any(e.name == name for e in entities):
                entities.append(
                    Entity(
                        id=f"rule_{entity_id}",
                        name=name,
                        type=EntityType.OTHER,
                    )
                )
                entity_id += 1

        return entities


class LLMEntityExtractor(EntityExtractor):
    """
    LLM-based entity extractor.

    Uses an LLM to extract and classify entities.
    """

    EXTRACTION_PROMPT = """Extract named entities from the following text.
For each entity, identify:
1. The entity name
2. The entity type (person, organization, location, date, event, product, concept)

Text:
{text}

Output format (one entity per line):
ENTITY: <name> | TYPE: <type>

Entities:"""

    def __init__(self, llm: Optional[Callable[[str], str]] = None):
        """Initialize extractor."""
        self.llm = llm
        self._fallback = RuleBasedEntityExtractor()

    async def extract(self, text: str) -> List[Entity]:
        """Extract entities using LLM."""
        if self.llm is None:
            return await self._fallback.extract(text)

        prompt = self.EXTRACTION_PROMPT.format(text=text[:2000])

        try:
            if asyncio.iscoroutinefunction(self.llm):
                response = await self.llm(prompt)
            else:
                response = self.llm(prompt)

            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return await self._fallback.extract(text)

    def _parse_response(self, response: str) -> List[Entity]:
        """Parse LLM response."""
        entities = []
        entity_id = 0

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("ENTITY:"):
                parts = line.split("|")
                if len(parts) >= 2:
                    name = parts[0].replace("ENTITY:", "").strip()
                    type_str = parts[1].replace("TYPE:", "").strip().lower()

                    entity_type = EntityType.OTHER
                    for et in EntityType:
                        if et.value == type_str:
                            entity_type = et
                            break

                    entities.append(
                        Entity(
                            id=f"llm_{entity_id}",
                            name=name,
                            type=entity_type,
                        )
                    )
                    entity_id += 1

        return entities


# =============================================================================
# Graph-RAG Retriever
# =============================================================================


class GraphRAG:
    """
    Graph-enhanced RAG retriever.

    Combines knowledge graph traversal with vector retrieval.

    Example:
        >>> graph_rag = GraphRAG(extractor, graph, retriever)
        >>> result = await graph_rag.retrieve("Who founded Microsoft?", k=5)
    """

    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        base_retriever: Optional[RetrieverProtocol] = None,
        embedder: Optional[EmbedderProtocol] = None,
        graph_weight: float = 0.3,
    ):
        """
        Initialize Graph-RAG.

        Args:
            entity_extractor: Entity extractor
            knowledge_graph: Knowledge graph
            base_retriever: Base vector retriever
            embedder: Text embedder
            graph_weight: Weight for graph-based scores
        """
        self.entity_extractor = entity_extractor or RuleBasedEntityExtractor()
        self.graph = knowledge_graph or KnowledgeGraph()
        self.retriever = base_retriever
        self.embedder = embedder
        self.graph_weight = graph_weight

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        expand_entities: bool = True,
        subgraph_depth: int = 2,
    ) -> GraphRAGResult:
        """
        Retrieve documents with graph enhancement.

        Args:
            query: Query string
            k: Number of documents to retrieve
            expand_entities: Whether to expand entities via graph
            subgraph_depth: Depth for subgraph extraction

        Returns:
            GraphRAGResult with documents and context
        """
        # Extract entities from query
        query_entities = await self.entity_extractor.extract(query)

        # Find matching entities in graph
        matched_entities: List[Entity] = []
        for qe in query_entities:
            matches = self.graph.get_entity_by_name(qe.name)
            matched_entities.extend(matches)

        # Get subgraph for matched entities
        subgraph: Optional[Subgraph] = None
        if matched_entities and expand_entities:
            # Get subgraph for first matched entity
            subgraph = self.graph.get_subgraph(
                matched_entities[0].id,
                depth=subgraph_depth,
            )

        # Base retrieval
        documents: List[Document] = []
        if self.retriever:
            # Expand query with entity context
            expanded_query = self._expand_query(query, matched_entities)
            docs = await self.retriever.retrieve(expanded_query, k=k * 2)

            # Convert to Document type if needed
            for doc in docs:
                if isinstance(doc, Document):
                    documents.append(doc)
                else:
                    documents.append(
                        Document(
                            id=str(hash(str(doc))),
                            content=str(doc),
                        )
                    )

        # Extract entities from documents
        for doc in documents:
            doc_entities = await self.entity_extractor.extract(doc.content)
            doc.entities = doc_entities

        # Rerank with entity overlap
        documents = self._rerank_with_entities(
            documents,
            matched_entities,
            subgraph.entities if subgraph else [],
        )

        # Take top k
        documents = documents[:k]

        # Build context string
        context_parts = []
        if subgraph:
            context_parts.append("Knowledge Graph Context:")
            context_parts.append(subgraph.to_context_string())
            context_parts.append("")

        context_parts.append("Retrieved Documents:")
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"{i}. {doc.content[:500]}")

        return GraphRAGResult(
            documents=documents,
            entities=matched_entities,
            subgraph=subgraph,
            query_entities=query_entities,
            context_string="\n".join(context_parts),
        )

    async def index_document(self, doc: Document) -> None:
        """
        Index a document in the knowledge graph.

        Args:
            doc: Document to index
        """
        # Extract entities
        entities = await self.entity_extractor.extract(doc.content)

        # Add entities to graph
        for entity in entities:
            entity.source_docs.append(doc.id)
            self.graph.add_entity(entity)

        # Add co-occurrence relations
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                self.graph.add_relation(
                    Relation(
                        source=e1.id,
                        target=e2.id,
                        type=RelationType.RELATED_TO,
                        source_doc=doc.id,
                    )
                )

    def _expand_query(
        self,
        query: str,
        entities: List[Entity],
    ) -> str:
        """Expand query with entity information."""
        if not entities:
            return query

        entity_names = [e.name for e in entities[:3]]
        expansion = " ".join(entity_names)
        return f"{query} {expansion}"

    def _rerank_with_entities(
        self,
        documents: List[Document],
        query_entities: List[Entity],
        graph_entities: List[Entity],
    ) -> List[Document]:
        """Rerank documents based on entity overlap."""
        query_entity_names = {e.name.lower() for e in query_entities}
        graph_entity_names = {e.name.lower() for e in graph_entities}
        all_entity_names = query_entity_names | graph_entity_names

        for doc in documents:
            doc_entity_names = {e.name.lower() for e in doc.entities}

            # Calculate entity overlap
            overlap = len(doc_entity_names & all_entity_names)
            entity_score = overlap / max(len(all_entity_names), 1)

            # Combine with base score
            doc.score = (1 - self.graph_weight) * doc.score + self.graph_weight * entity_score

        # Sort by combined score
        return sorted(documents, key=lambda d: d.score, reverse=True)


# =============================================================================
# Graph Builder
# =============================================================================


class GraphBuilder:
    """
    Builds knowledge graph from documents.

    Extracts entities and relations to construct a graph.
    """

    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[Callable[[str, Entity, Entity], Optional[Relation]]] = None,
    ):
        """Initialize builder."""
        self.entity_extractor = entity_extractor or RuleBasedEntityExtractor()
        self.relation_extractor = relation_extractor

    async def build_from_documents(
        self,
        documents: List[Document],
    ) -> KnowledgeGraph:
        """
        Build graph from documents.

        Args:
            documents: Documents to process

        Returns:
            Constructed knowledge graph
        """
        graph = KnowledgeGraph()

        for doc in documents:
            # Extract entities
            entities = await self.entity_extractor.extract(doc.content)

            # Add to graph
            for entity in entities:
                entity.source_docs.append(doc.id)
                graph.add_entity(entity)

            # Extract/infer relations
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    if self.relation_extractor:
                        rel = self.relation_extractor(doc.content, e1, e2)
                        if rel:
                            graph.add_relation(rel)
                    else:
                        # Default to co-occurrence relation
                        graph.add_relation(
                            Relation(
                                source=e1.id,
                                target=e2.id,
                                type=RelationType.RELATED_TO,
                                source_doc=doc.id,
                            )
                        )

        return graph


# =============================================================================
# Convenience Functions
# =============================================================================


def create_graph_rag(
    retriever: Optional[RetrieverProtocol] = None,
    graph_weight: float = 0.3,
) -> GraphRAG:
    """
    Create a Graph-RAG instance.

    Args:
        retriever: Base retriever
        graph_weight: Weight for graph scores

    Returns:
        GraphRAG instance
    """
    return GraphRAG(
        base_retriever=retriever,
        graph_weight=graph_weight,
    )


async def extract_entities(text: str) -> List[Entity]:
    """
    Extract entities from text.

    Args:
        text: Input text

    Returns:
        List of extracted entities
    """
    extractor = RuleBasedEntityExtractor()
    return await extractor.extract(text)
