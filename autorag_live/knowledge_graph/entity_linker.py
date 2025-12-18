"""
Entity linking module for AutoRAG-Live.

Provides entity extraction and linking to knowledge bases
for enhanced retrieval and understanding.

Features:
- Named entity recognition
- Entity disambiguation
- Knowledge base linking
- Entity caching
- Relationship extraction
- Confidence scoring

Example usage:
    >>> linker = EntityLinker()
    >>> entities = linker.link("Apple released the new iPhone in Cupertino")
    >>> for entity in entities:
    ...     print(f"{entity.text} -> {entity.kb_id} ({entity.entity_type})")
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of named entities."""
    
    PERSON = auto()
    ORGANIZATION = auto()
    LOCATION = auto()
    DATE = auto()
    TIME = auto()
    MONEY = auto()
    PERCENT = auto()
    PRODUCT = auto()
    EVENT = auto()
    WORK_OF_ART = auto()
    LAW = auto()
    LANGUAGE = auto()
    CONCEPT = auto()
    UNKNOWN = auto()


class LinkingStatus(Enum):
    """Status of entity linking."""
    
    LINKED = auto()
    CANDIDATE = auto()
    NOT_FOUND = auto()
    AMBIGUOUS = auto()


@dataclass
class EntityMention:
    """An entity mention in text."""
    
    text: str
    entity_type: EntityType
    
    # Position
    start: int
    end: int
    
    # Confidence
    confidence: float = 1.0
    
    # Context
    context: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeBaseEntry:
    """An entry in the knowledge base."""
    
    kb_id: str
    name: str
    entity_type: EntityType
    
    # Aliases
    aliases: List[str] = field(default_factory=list)
    
    # Description
    description: str = ""
    
    # Relationships
    related_entities: List[str] = field(default_factory=list)
    
    # External IDs
    wikidata_id: Optional[str] = None
    wikipedia_url: Optional[str] = None
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkedEntity:
    """An entity linked to knowledge base."""
    
    mention: EntityMention
    kb_entry: Optional[KnowledgeBaseEntry]
    
    # Linking info
    status: LinkingStatus = LinkingStatus.NOT_FOUND
    linking_score: float = 0.0
    
    # Candidates
    candidates: List[Tuple[KnowledgeBaseEntry, float]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def text(self) -> str:
        return self.mention.text
    
    @property
    def entity_type(self) -> EntityType:
        return self.mention.entity_type
    
    @property
    def kb_id(self) -> Optional[str]:
        return self.kb_entry.kb_id if self.kb_entry else None
    
    @property
    def is_linked(self) -> bool:
        return self.status == LinkingStatus.LINKED


@dataclass
class Relationship:
    """A relationship between entities."""
    
    source: LinkedEntity
    target: LinkedEntity
    relation_type: str
    
    # Confidence
    confidence: float = 1.0
    
    # Context
    context: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEntityRecognizer(ABC):
    """Base class for entity recognition."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Recognizer name."""
        pass
    
    @abstractmethod
    def recognize(self, text: str) -> List[EntityMention]:
        """Recognize entities in text."""
        pass


class PatternEntityRecognizer(BaseEntityRecognizer):
    """Pattern-based entity recognition."""
    
    # Common patterns
    PATTERNS = {
        EntityType.DATE: [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',
        ],
        EntityType.TIME: [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        ],
        EntityType.MONEY: [
            r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',
            r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)\b',
        ],
        EntityType.PERCENT: [
            r'\b\d+(?:\.\d+)?%\b',
        ],
        EntityType.PERSON: [
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        ],
    }
    
    def __init__(
        self,
        custom_patterns: Optional[Dict[EntityType, List[str]]] = None,
    ):
        """
        Initialize recognizer.
        
        Args:
            custom_patterns: Additional patterns
        """
        self.patterns = self.PATTERNS.copy()
        if custom_patterns:
            for entity_type, patterns in custom_patterns.items():
                if entity_type in self.patterns:
                    self.patterns[entity_type].extend(patterns)
                else:
                    self.patterns[entity_type] = patterns
        
        # Compile patterns
        self._compiled = {
            entity_type: [re.compile(p) for p in patterns]
            for entity_type, patterns in self.patterns.items()
        }
    
    @property
    def name(self) -> str:
        return "pattern"
    
    def recognize(self, text: str) -> List[EntityMention]:
        """Recognize entities using patterns."""
        mentions = []
        
        for entity_type, patterns in self._compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    mentions.append(EntityMention(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                    ))
        
        return self._deduplicate(mentions)
    
    def _deduplicate(
        self,
        mentions: List[EntityMention],
    ) -> List[EntityMention]:
        """Remove overlapping mentions."""
        if not mentions:
            return []
        
        # Sort by start position
        sorted_mentions = sorted(mentions, key=lambda m: (m.start, -len(m.text)))
        
        result = []
        last_end = -1
        
        for mention in sorted_mentions:
            if mention.start >= last_end:
                result.append(mention)
                last_end = mention.end
        
        return result


class NEREntityRecognizer(BaseEntityRecognizer):
    """NER-based entity recognition (placeholder for ML models)."""
    
    # Type mapping from common NER labels
    TYPE_MAP = {
        'PER': EntityType.PERSON,
        'PERSON': EntityType.PERSON,
        'ORG': EntityType.ORGANIZATION,
        'ORGANIZATION': EntityType.ORGANIZATION,
        'LOC': EntityType.LOCATION,
        'LOCATION': EntityType.LOCATION,
        'GPE': EntityType.LOCATION,
        'DATE': EntityType.DATE,
        'TIME': EntityType.TIME,
        'MONEY': EntityType.MONEY,
        'PERCENT': EntityType.PERCENT,
        'PRODUCT': EntityType.PRODUCT,
        'EVENT': EntityType.EVENT,
        'WORK_OF_ART': EntityType.WORK_OF_ART,
        'LAW': EntityType.LAW,
        'LANGUAGE': EntityType.LANGUAGE,
    }
    
    def __init__(
        self,
        model: str = "en_core_web_sm",
    ):
        """
        Initialize recognizer.
        
        Args:
            model: NER model name
        """
        self.model_name = model
        self._nlp = None
    
    @property
    def name(self) -> str:
        return "ner"
    
    def _load_model(self) -> None:
        """Load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.model_name)
            except ImportError:
                logger.warning("spaCy not installed, using fallback")
                self._nlp = None
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                self._nlp = None
    
    def recognize(self, text: str) -> List[EntityMention]:
        """Recognize entities using NER."""
        self._load_model()
        
        if self._nlp is None:
            # Fallback to pattern-based
            return PatternEntityRecognizer().recognize(text)
        
        doc = self._nlp(text)
        mentions = []
        
        for ent in doc.ents:
            entity_type = self.TYPE_MAP.get(ent.label_, EntityType.UNKNOWN)
            
            mentions.append(EntityMention(
                text=ent.text,
                entity_type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.9,
                metadata={'label': ent.label_},
            ))
        
        return mentions


class KnowledgeBase:
    """In-memory knowledge base."""
    
    def __init__(self):
        """Initialize knowledge base."""
        self._entries: Dict[str, KnowledgeBaseEntry] = {}
        self._name_index: Dict[str, Set[str]] = {}  # name -> kb_ids
        self._type_index: Dict[EntityType, Set[str]] = {}
    
    def add_entry(self, entry: KnowledgeBaseEntry) -> None:
        """Add entry to knowledge base."""
        self._entries[entry.kb_id] = entry
        
        # Index by name
        self._index_name(entry.name.lower(), entry.kb_id)
        for alias in entry.aliases:
            self._index_name(alias.lower(), entry.kb_id)
        
        # Index by type
        if entry.entity_type not in self._type_index:
            self._type_index[entry.entity_type] = set()
        self._type_index[entry.entity_type].add(entry.kb_id)
    
    def _index_name(self, name: str, kb_id: str) -> None:
        """Index entry by name."""
        if name not in self._name_index:
            self._name_index[name] = set()
        self._name_index[name].add(kb_id)
    
    def get_entry(self, kb_id: str) -> Optional[KnowledgeBaseEntry]:
        """Get entry by ID."""
        return self._entries.get(kb_id)
    
    def search_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> List[KnowledgeBaseEntry]:
        """Search entries by name."""
        name_lower = name.lower()
        
        # Exact match
        kb_ids = self._name_index.get(name_lower, set())
        
        # Filter by type if specified
        if entity_type and entity_type in self._type_index:
            kb_ids = kb_ids & self._type_index[entity_type]
        
        return [self._entries[kb_id] for kb_id in kb_ids]
    
    def search_fuzzy(
        self,
        name: str,
        threshold: float = 0.8,
    ) -> List[Tuple[KnowledgeBaseEntry, float]]:
        """Fuzzy search by name."""
        results = []
        name_lower = name.lower()
        
        for indexed_name, kb_ids in self._name_index.items():
            score = self._similarity(name_lower, indexed_name)
            if score >= threshold:
                for kb_id in kb_ids:
                    results.append((self._entries[kb_id], score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        # Simple Jaccard similarity on character n-grams
        n = 2
        
        def ngrams(s: str) -> Set[str]:
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ng1 = ngrams(s1)
        ng2 = ngrams(s2)
        
        if not ng1 or not ng2:
            return 0.0
        
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        
        return intersection / union if union > 0 else 0.0
    
    @property
    def size(self) -> int:
        """Get number of entries."""
        return len(self._entries)


class EntityCache:
    """Cache for entity linking results."""
    
    def __init__(
        self,
        max_size: int = 10000,
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum cache entries
        """
        self.max_size = max_size
        self._cache: Dict[str, LinkedEntity] = {}
    
    def _make_key(
        self,
        text: str,
        entity_type: EntityType,
    ) -> str:
        """Generate cache key."""
        content = f"{entity_type.name}:{text.lower()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        entity_type: EntityType,
    ) -> Optional[LinkedEntity]:
        """Get cached entity."""
        key = self._make_key(text, entity_type)
        return self._cache.get(key)
    
    def set(
        self,
        entity: LinkedEntity,
    ) -> None:
        """Cache entity."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entries
            keys_to_remove = list(self._cache.keys())[:len(self._cache) // 4]
            for key in keys_to_remove:
                del self._cache[key]
        
        key = self._make_key(entity.text, entity.entity_type)
        self._cache[key] = entity
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


class EntityLinker:
    """
    Main entity linking interface.
    
    Example:
        >>> linker = EntityLinker()
        >>> 
        >>> # Add knowledge base entries
        >>> linker.add_kb_entry(KnowledgeBaseEntry(
        ...     kb_id="Q312",
        ...     name="Apple Inc.",
        ...     entity_type=EntityType.ORGANIZATION,
        ...     aliases=["Apple", "Apple Computer"],
        ... ))
        >>> 
        >>> # Link entities
        >>> entities = linker.link("Apple released a new product")
        >>> for entity in entities:
        ...     if entity.is_linked:
        ...         print(f"{entity.text} -> {entity.kb_id}")
    """
    
    def __init__(
        self,
        recognizer: Optional[BaseEntityRecognizer] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        use_cache: bool = True,
        linking_threshold: float = 0.7,
    ):
        """
        Initialize entity linker.
        
        Args:
            recognizer: Entity recognizer
            knowledge_base: Knowledge base for linking
            use_cache: Enable caching
            linking_threshold: Minimum score for linking
        """
        self.recognizer = recognizer or NEREntityRecognizer()
        self.kb = knowledge_base or KnowledgeBase()
        self.linking_threshold = linking_threshold
        
        self._cache = EntityCache() if use_cache else None
    
    def add_kb_entry(self, entry: KnowledgeBaseEntry) -> None:
        """Add entry to knowledge base."""
        self.kb.add_entry(entry)
    
    def recognize(self, text: str) -> List[EntityMention]:
        """Recognize entities in text."""
        return self.recognizer.recognize(text)
    
    def link_mention(
        self,
        mention: EntityMention,
    ) -> LinkedEntity:
        """
        Link a single entity mention.
        
        Args:
            mention: Entity mention
            
        Returns:
            LinkedEntity
        """
        # Check cache
        if self._cache:
            cached = self._cache.get(mention.text, mention.entity_type)
            if cached:
                return cached
        
        # Search knowledge base
        candidates = []
        
        # Exact match
        exact_matches = self.kb.search_by_name(
            mention.text,
            mention.entity_type,
        )
        for entry in exact_matches:
            candidates.append((entry, 1.0))
        
        # Fuzzy match if no exact
        if not candidates:
            fuzzy_matches = self.kb.search_fuzzy(
                mention.text,
                threshold=self.linking_threshold,
            )
            candidates.extend(fuzzy_matches)
        
        # Create linked entity
        if candidates:
            # Best match
            best_entry, best_score = candidates[0]
            
            if best_score >= self.linking_threshold:
                status = LinkingStatus.LINKED
                if len(candidates) > 1 and candidates[1][1] > self.linking_threshold:
                    status = LinkingStatus.AMBIGUOUS
            else:
                status = LinkingStatus.CANDIDATE
                best_entry = None
            
            linked = LinkedEntity(
                mention=mention,
                kb_entry=best_entry,
                status=status,
                linking_score=best_score,
                candidates=candidates[:5],
            )
        else:
            linked = LinkedEntity(
                mention=mention,
                kb_entry=None,
                status=LinkingStatus.NOT_FOUND,
            )
        
        # Cache result
        if self._cache:
            self._cache.set(linked)
        
        return linked
    
    def link(self, text: str) -> List[LinkedEntity]:
        """
        Recognize and link entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of LinkedEntity
        """
        mentions = self.recognize(text)
        return [self.link_mention(m) for m in mentions]
    
    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[LinkedEntity]] = None,
    ) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            text: Input text
            entities: Pre-linked entities
            
        Returns:
            List of Relationship
        """
        if entities is None:
            entities = self.link(text)
        
        relationships = []
        
        # Simple co-occurrence based relationships
        # Real implementation would use dependency parsing
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities are in same sentence
                # Simplified: check if within 100 characters
                if abs(entity1.mention.start - entity2.mention.start) < 100:
                    # Extract context between entities
                    start = min(entity1.mention.end, entity2.mention.end)
                    end = max(entity1.mention.start, entity2.mention.start)
                    context = text[start:end].strip()
                    
                    # Simple relation detection
                    relation_type = self._detect_relation(context)
                    
                    if relation_type:
                        relationships.append(Relationship(
                            source=entity1,
                            target=entity2,
                            relation_type=relation_type,
                            context=context,
                        ))
        
        return relationships
    
    def _detect_relation(self, context: str) -> Optional[str]:
        """Detect relation type from context."""
        context_lower = context.lower()
        
        # Simple pattern matching
        patterns = {
            'works_for': ['works for', 'employed by', 'ceo of'],
            'located_in': ['in', 'at', 'based in', 'located in'],
            'founded': ['founded', 'established', 'created'],
            'acquired': ['acquired', 'bought', 'purchased'],
            'part_of': ['part of', 'member of', 'belongs to'],
        }
        
        for relation, keywords in patterns.items():
            for keyword in keywords:
                if keyword in context_lower:
                    return relation
        
        return None
    
    def clear_cache(self) -> None:
        """Clear entity cache."""
        if self._cache:
            self._cache.clear()


# Convenience functions

def link_entities(
    text: str,
    recognizer: str = "pattern",
) -> List[LinkedEntity]:
    """
    Quick entity linking.
    
    Args:
        text: Input text
        recognizer: Recognizer type
        
    Returns:
        List of linked entities
    """
    if recognizer == "pattern":
        rec = PatternEntityRecognizer()
    else:
        rec = NEREntityRecognizer()
    
    linker = EntityLinker(recognizer=rec)
    return linker.link(text)


def recognize_entities(
    text: str,
) -> List[EntityMention]:
    """
    Quick entity recognition.
    
    Args:
        text: Input text
        
    Returns:
        List of entity mentions
    """
    recognizer = PatternEntityRecognizer()
    return recognizer.recognize(text)


def create_kb_entry(
    kb_id: str,
    name: str,
    entity_type: str,
    aliases: Optional[List[str]] = None,
    **kwargs,
) -> KnowledgeBaseEntry:
    """
    Create knowledge base entry.
    
    Args:
        kb_id: Unique ID
        name: Entity name
        entity_type: Entity type string
        aliases: Alternative names
        **kwargs: Additional properties
        
    Returns:
        KnowledgeBaseEntry
    """
    type_map = {
        'person': EntityType.PERSON,
        'organization': EntityType.ORGANIZATION,
        'location': EntityType.LOCATION,
        'product': EntityType.PRODUCT,
        'event': EntityType.EVENT,
    }
    
    return KnowledgeBaseEntry(
        kb_id=kb_id,
        name=name,
        entity_type=type_map.get(entity_type.lower(), EntityType.UNKNOWN),
        aliases=aliases or [],
        **kwargs,
    )
