"""
Knowledge graph integration for semantic understanding and multi-hop reasoning.

Supports entity extraction, relation discovery, and graph-based query expansion.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""

    name: str
    entity_type: str  # "person", "location", "concept", "organization"
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    mentions: List[str] = field(default_factory=list)

    def add_mention(self, mention: str) -> None:
        """Add a mention of this entity."""
        if mention not in self.mentions:
            self.mentions.append(mention)

    def __hash__(self) -> int:
        """Make entity hashable."""
        return hash((self.name, self.entity_type))

    def __eq__(self, other: Any) -> bool:
        """Check entity equality."""
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.entity_type == other.entity_type


@dataclass
class Relation:
    """Represents a relation between entities."""

    source: Entity
    relation_type: str
    target: Entity
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)

    def add_evidence(self, text: str) -> None:
        """Add evidence for this relation."""
        if text not in self.evidence:
            self.evidence.append(text)


@dataclass
class KnowledgeGraphNode:
    """Node in the knowledge graph."""

    entity: Entity
    relations_out: List[Relation] = field(default_factory=list)
    relations_in: List[Relation] = field(default_factory=list)

    def get_neighbors(self, max_hops: int = 1) -> Set[Entity]:
        """Get neighboring entities within max hops (deque BFS)."""
        neighbors: Set[Entity] = set()
        to_visit: deque[tuple[Entity, int]] = deque([(self.entity, 0)])
        visited: Set[Entity] = set()

        while to_visit:
            current, hops = to_visit.popleft()  # O(1) vs list.pop(0) O(n)

            if current in visited:
                continue

            visited.add(current)
            neighbors.add(current)

            if hops < max_hops:
                node = self._find_node(current)
                if node:
                    for rel in node.relations_out:
                        if rel.target not in visited:
                            to_visit.append((rel.target, hops + 1))
                    for rel in node.relations_in:
                        if rel.source not in visited:
                            to_visit.append((rel.source, hops + 1))

        return neighbors

    def _find_node(self, entity: Entity) -> Optional[KnowledgeGraphNode]:
        """Find node for entity."""
        # This would be implemented by KnowledgeGraph
        return None


class KnowledgeGraph:
    """
    Semantic knowledge graph for entity and relation understanding.

    Enables multi-hop reasoning, query expansion, and semantic navigation.
    """

    def __init__(self):
        """Initialize knowledge graph."""
        self.logger = logging.getLogger("KnowledgeGraph")
        self.nodes: Dict[Tuple[str, str], KnowledgeGraphNode] = {}
        self.entities: Dict[Tuple[str, str], Entity] = {}
        self.relations: List[Relation] = []

        # O(1) name → key lookup (avoids linear scan on every find_path / context call)
        self._name_index: Dict[str, Tuple[str, str]] = {}

        # Entity type patterns
        self.entity_patterns = {
            "person": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            "location": r"\b(?:in |at |from )([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b",
            "concept": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b",
            "organization": r"\b(?:the )([A-Z][a-z0-9]+(?:\s[A-Z][a-z0-9]+)*)\b",
        }

    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Add entity to graph."""
        key = (name, entity_type)

        if key in self.entities:
            entity = self.entities[key]
            if description:
                entity.description = description
            if properties:
                entity.properties.update(properties)
        else:
            entity = Entity(
                name=name,
                entity_type=entity_type,
                description=description,
                properties=properties or {},
            )
            self.entities[key] = entity
            self._name_index[name] = key  # O(1) name lookup

            # Create graph node
            node = KnowledgeGraphNode(entity)
            self.nodes[key] = node

        return entity

    def add_relation(
        self,
        source_name: str,
        relation_type: str,
        target_name: str,
        source_type: str = "concept",
        target_type: str = "concept",
        confidence: float = 0.5,
        evidence: Optional[str] = None,
    ) -> Relation:
        """Add relation between entities."""
        # Ensure entities exist
        source = self.add_entity(source_name, source_type)
        target = self.add_entity(target_name, target_type)

        # Create relation
        relation = Relation(
            source=source,
            relation_type=relation_type,
            target=target,
            confidence=confidence,
        )

        if evidence:
            relation.add_evidence(evidence)

        self.relations.append(relation)

        # Update nodes
        source_key = (source_name, source_type)
        target_key = (target_name, target_type)

        if source_key in self.nodes:
            self.nodes[source_key].relations_out.append(relation)
        if target_key in self.nodes:
            self.nodes[target_key].relations_in.append(relation)

        return relation

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        extracted = []

        # Simple regex-based extraction
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group(0).strip()
                if entity_text:
                    entity = self.add_entity(
                        entity_text, entity_type, properties={"source": "text"}
                    )
                    entity.add_mention(entity_text)
                    if entity not in extracted:
                        extracted.append(entity)

        return extracted

    def discover_relations(self, text: str, confidence_threshold: float = 0.3) -> List[Relation]:
        """Discover relations from text."""
        discovered = []
        entities = self.extract_entities(text)

        # Simple pattern-based relation discovery
        relation_patterns = {
            "located_in": r"(\w+) (?:is |located |situated )(?:in |at )(\w+)",
            "works_for": r"(\w+) (?:works|works_for) (\w+)",
            "has_property": r"(\w+) (?:is |has )(\w+)",
            "part_of": r"(\w+) (?:is part of|is part|belongs to) (\w+)",
        }

        for rel_type, pattern in relation_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_text, target_text = match.groups()

                # Find corresponding entities
                source = next(
                    (e for e in entities if source_text.lower() in e.name.lower()),
                    None,
                )
                target = next(
                    (e for e in entities if target_text.lower() in e.name.lower()),
                    None,
                )

                if source and target:
                    relation = self.add_relation(
                        source.name,
                        rel_type,
                        target.name,
                        source_type=source.entity_type,
                        target_type=target.entity_type,
                        confidence=confidence_threshold,
                        evidence=match.group(0),
                    )
                    discovered.append(relation)

        return discovered

    def expand_query(self, query: str, hops: int = 1) -> List[str]:
        """Expand query using knowledge graph."""
        expanded = [query]

        # Extract entities from query
        entities = self.extract_entities(query)

        # Get related entities
        for entity in entities:
            key = (entity.name, entity.entity_type)
            if key in self.nodes:
                node = self.nodes[key]

                # Get neighbors
                neighbors = self._get_neighbors(node, hops)
                for neighbor in neighbors:
                    # Create expanded query
                    expanded_query = query.replace(entity.name, neighbor.name)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)

        return expanded

    def find_path(self, source_name: str, target_name: str) -> Optional[List[Entity]]:
        """Find path between two entities (BFS with O(1) name lookup)."""
        source_key = self._name_index.get(source_name)
        target_key = self._name_index.get(target_name)

        if not source_key or not target_key:
            return None

        source = self.entities[source_key]
        target = self.entities[target_key]

        # BFS
        queue: deque[tuple[Entity, list[Entity]]] = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            # Get neighbors
            current_key = (current.name, current.entity_type)
            if current_key in self.nodes:
                node = self.nodes[current_key]

                for rel in node.relations_out:
                    if rel.target not in visited:
                        visited.add(rel.target)
                        queue.append((rel.target, path + [rel.target]))

        return None

    def get_entity_context(self, entity_name: str, hops: int = 1) -> Dict[str, Any]:
        """Get context around an entity (O(1) name lookup)."""
        entity_key = self._name_index.get(entity_name)

        if not entity_key:
            return {}

        entity = self.entities[entity_key]
        node = self.nodes[entity_key]

        # Collect related info
        context = {
            "entity": entity.name,
            "type": entity.entity_type,
            "description": entity.description,
            "properties": entity.properties,
            "mentions": entity.mentions,
            "related": [],
        }

        # Add related entities
        for rel in node.relations_out[:5]:
            context["related"].append(
                {
                    "type": rel.relation_type,
                    "target": rel.target.name,
                    "confidence": rel.confidence,
                }
            )

        for rel in node.relations_in[:5]:
            context["related"].append(
                {
                    "type": f"inverse_{rel.relation_type}",
                    "source": rel.source.name,
                    "confidence": rel.confidence,
                }
            )

        return context

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge graph."""
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "entity_types": list(set(e[1] for e in self.entities.keys())),
            "relation_types": list(set(r.relation_type for r in self.relations)),
        }

    def _get_neighbors(self, node: KnowledgeGraphNode, max_hops: int) -> Set[Entity]:
        """Get neighboring entities within max hops.

        Uses collections.deque for O(1) popleft instead of list.pop(0) O(n).
        Edges are traversed in descending confidence order so that
        high-quality relations are explored first.
        """
        neighbors: Set[Entity] = set()
        to_visit: deque[tuple[Entity, int]] = deque([(node.entity, 0)])
        visited: Set[Entity] = set()

        while to_visit:
            current, hops = to_visit.popleft()  # O(1)

            if current in visited:
                continue

            visited.add(current)

            if hops > 0:
                neighbors.add(current)

            if hops < max_hops:
                current_key = (current.name, current.entity_type)
                if current_key in self.nodes:
                    current_node = self.nodes[current_key]

                    # Traverse highest-confidence edges first
                    for rel in sorted(
                        current_node.relations_out,
                        key=lambda r: r.confidence,
                        reverse=True,
                    ):
                        if rel.target not in visited:
                            to_visit.append((rel.target, hops + 1))

                    for rel in sorted(
                        current_node.relations_in,
                        key=lambda r: r.confidence,
                        reverse=True,
                    ):
                        if rel.source not in visited:
                            to_visit.append((rel.source, hops + 1))

        return neighbors
