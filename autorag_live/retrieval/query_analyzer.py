"""
Query analysis utilities for AutoRAG-Live.

Provides deep query understanding including intent detection,
entity extraction, complexity analysis, and query optimization.

Features:
- Intent classification
- Entity extraction
- Query complexity analysis
- Query type detection
- Temporal analysis
- Query decomposition

Example usage:
    >>> analyzer = QueryAnalyzer()
    >>> analysis = analyzer.analyze("What is the capital of France?")
    >>> print(analysis.intent)
    >>> print(analysis.entities)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Query intent types."""
    
    FACTUAL = "factual"           # What, when, where facts
    EXPLANATORY = "explanatory"   # Why, how explanations
    COMPARATIVE = "comparative"   # Compare X and Y
    PROCEDURAL = "procedural"     # How to do something
    DEFINITIONAL = "definitional" # What is X
    CAUSAL = "causal"            # Why does X cause Y
    OPINION = "opinion"          # What do you think
    LIST = "list"                # List of items
    YES_NO = "yes_no"            # Yes/no questions
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    """Query structure types."""
    
    SIMPLE = "simple"             # Single, direct question
    COMPLEX = "complex"           # Multi-part question
    COMPOUND = "compound"         # Multiple related questions
    CONVERSATIONAL = "conversational"  # Follow-up/reference
    KEYWORD = "keyword"           # Keyword search style
    NATURAL = "natural"           # Natural language


class EntityType(str, Enum):
    """Entity types."""
    
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    NUMBER = "number"
    PRODUCT = "product"
    EVENT = "event"
    CONCEPT = "concept"
    TECHNICAL = "technical"
    OTHER = "other"


class TemporalReference(str, Enum):
    """Temporal reference types."""
    
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    NONE = "none"


@dataclass
class Entity:
    """An extracted entity."""
    
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryComponent:
    """A component of a decomposed query."""
    
    text: str
    type: str
    importance: float = 1.0
    dependencies: List[int] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    
    query: str
    
    # Intent and type
    intent: QueryIntent = QueryIntent.UNKNOWN
    intent_confidence: float = 0.0
    query_type: QueryType = QueryType.SIMPLE
    
    # Entities
    entities: List[Entity] = field(default_factory=list)
    
    # Structure
    keywords: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    components: List[QueryComponent] = field(default_factory=list)
    
    # Temporal
    temporal_reference: TemporalReference = TemporalReference.NONE
    temporal_expressions: List[str] = field(default_factory=list)
    
    # Complexity
    complexity_score: float = 0.0
    estimated_context_length: int = 1
    
    # Quality
    is_well_formed: bool = True
    clarity_score: float = 1.0
    specificity_score: float = 0.5
    
    # Suggestions
    rewritten_query: Optional[str] = None
    sub_queries: List[str] = field(default_factory=list)
    
    # Metadata
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_simple(self) -> bool:
        """Check if query is simple."""
        return self.complexity_score < 0.3
    
    @property
    def is_complex(self) -> bool:
        """Check if query is complex."""
        return self.complexity_score >= 0.7
    
    @property
    def entity_count(self) -> int:
        """Get number of entities."""
        return len(self.entities)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get entities of specific type."""
        return [e for e in self.entities if e.type == entity_type]


class IntentClassifier:
    """Classify query intent."""
    
    # Intent patterns
    INTENT_PATTERNS = {
        QueryIntent.DEFINITIONAL: [
            r'^what\s+is\s+',
            r'^define\s+',
            r'^what\s+are\s+',
            r'^what\s+does\s+.+\s+mean',
        ],
        QueryIntent.EXPLANATORY: [
            r'^why\s+',
            r'^how\s+does\s+',
            r'^explain\s+',
            r'^can\s+you\s+explain\s+',
        ],
        QueryIntent.PROCEDURAL: [
            r'^how\s+to\s+',
            r'^how\s+can\s+i\s+',
            r'^how\s+do\s+i\s+',
            r'^steps\s+to\s+',
            r'^guide\s+to\s+',
        ],
        QueryIntent.COMPARATIVE: [
            r'compare\s+',
            r'difference\s+between\s+',
            r'versus\s+',
            r'\s+vs\.?\s+',
            r'better\s+than\s+',
        ],
        QueryIntent.FACTUAL: [
            r'^what\s+',
            r'^when\s+',
            r'^where\s+',
            r'^who\s+',
            r'^which\s+',
        ],
        QueryIntent.LIST: [
            r'^list\s+',
            r'^give\s+me\s+.+\s+examples',
            r'^what\s+are\s+(?:some|the)\s+',
            r'^name\s+',
        ],
        QueryIntent.YES_NO: [
            r'^is\s+',
            r'^are\s+',
            r'^does\s+',
            r'^do\s+',
            r'^can\s+',
            r'^will\s+',
            r'^should\s+',
        ],
        QueryIntent.CAUSAL: [
            r'cause\s+',
            r'result\s+in\s+',
            r'lead\s+to\s+',
            r'effect\s+of\s+',
            r'because\s+',
        ],
    }
    
    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify query intent.
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (intent, confidence)
        """
        query_lower = query.lower().strip()
        
        # Check patterns
        best_intent = QueryIntent.UNKNOWN
        best_confidence = 0.0
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    confidence = self._calculate_confidence(
                        query_lower, pattern
                    )
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        # Default to factual if no match
        if best_intent == QueryIntent.UNKNOWN and best_confidence < 0.3:
            if '?' in query:
                best_intent = QueryIntent.FACTUAL
                best_confidence = 0.4
        
        return best_intent, min(best_confidence, 1.0)
    
    def _calculate_confidence(self, query: str, pattern: str) -> float:
        """Calculate pattern match confidence."""
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            return 0.0
        
        # Higher confidence for matches at start
        if match.start() == 0:
            return 0.9
        return 0.7


class EntityExtractor:
    """Extract entities from queries."""
    
    # Entity patterns
    ENTITY_PATTERNS = {
        EntityType.DATE: [
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\s+\d{1,2},?\s*\d{4}\b',
            r'\b\d{4}\b',
        ],
        EntityType.TIME: [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b',
        ],
        EntityType.NUMBER: [
            r'\b\d+(?:\.\d+)?(?:\s*%|percent)?\b',
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
        ],
        EntityType.LOCATION: [
            r'\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        ],
    }
    
    # Common named entities
    KNOWN_ENTITIES = {
        EntityType.ORGANIZATION: {
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta',
            'openai', 'anthropic', 'nvidia', 'tesla', 'ibm',
        },
        EntityType.TECHNICAL: {
            'python', 'javascript', 'java', 'rust', 'golang', 'typescript',
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi',
            'machine learning', 'deep learning', 'neural network', 'llm',
            'rag', 'transformer', 'gpt', 'bert', 'embedding',
        },
    }
    
    def extract(self, query: str) -> List[Entity]:
        """
        Extract entities from query.
        
        Args:
            query: Input query
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Pattern-based extraction
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    entities.append(Entity(
                        text=match.group(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                    ))
        
        # Dictionary-based extraction
        query_lower = query.lower()
        for entity_type, known in self.KNOWN_ENTITIES.items():
            for term in known:
                if term in query_lower:
                    start = query_lower.find(term)
                    entities.append(Entity(
                        text=query[start:start + len(term)],
                        type=entity_type,
                        start=start,
                        end=start + len(term),
                        confidence=0.9,
                        normalized=term,
                    ))
        
        # Remove overlapping entities
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping higher confidence."""
        if not entities:
            return []
        
        # Sort by start position and confidence
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start, -e.confidence)
        )
        
        result = []
        last_end = -1
        
        for entity in sorted_entities:
            if entity.start >= last_end:
                result.append(entity)
                last_end = entity.end
        
        return result


class KeywordExtractor:
    """Extract keywords from queries."""
    
    # Stop words
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we',
        'they', 'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    }
    
    def extract(
        self,
        query: str,
        max_keywords: int = 10,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract keywords and key phrases.
        
        Args:
            query: Input query
            max_keywords: Maximum keywords to return
            
        Returns:
            Tuple of (keywords, key_phrases)
        """
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words
        keywords = [
            w for w in words
            if w not in self.STOP_WORDS and len(w) > 2
        ]
        
        # Extract key phrases (n-grams)
        key_phrases = self._extract_ngrams(query, 2, 3)
        
        return keywords[:max_keywords], key_phrases[:5]
    
    def _extract_ngrams(
        self,
        query: str,
        min_n: int,
        max_n: int,
    ) -> List[str]:
        """Extract n-grams as potential key phrases."""
        words = query.lower().split()
        phrases = []
        
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                phrase_words = words[i:i + n]
                
                # Skip if starts/ends with stop word
                if phrase_words[0] in self.STOP_WORDS:
                    continue
                if phrase_words[-1] in self.STOP_WORDS:
                    continue
                
                phrase = ' '.join(phrase_words)
                phrases.append(phrase)
        
        return phrases


class ComplexityAnalyzer:
    """Analyze query complexity."""
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity.
        
        Args:
            query: Input query
            
        Returns:
            Complexity analysis dict
        """
        factors = {}
        
        # Word count factor
        word_count = len(query.split())
        factors['word_count'] = min(word_count / 20, 1.0)
        
        # Conjunction count
        conjunctions = len(re.findall(
            r'\b(?:and|or|but|however|although|while)\b',
            query.lower()
        ))
        factors['conjunctions'] = min(conjunctions / 3, 1.0)
        
        # Question count
        questions = query.count('?')
        factors['questions'] = min(questions / 2, 1.0)
        
        # Clause count (rough estimate)
        clauses = len(re.findall(r'[,;:]', query)) + 1
        factors['clauses'] = min(clauses / 4, 1.0)
        
        # Technical term density
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:tion|ment|ness|ity)\b',  # Abstract nouns
        ]
        technical_count = sum(
            len(re.findall(p, query))
            for p in technical_patterns
        )
        factors['technical'] = min(technical_count / 5, 1.0)
        
        # Calculate overall complexity
        weights = {
            'word_count': 0.2,
            'conjunctions': 0.25,
            'questions': 0.2,
            'clauses': 0.2,
            'technical': 0.15,
        }
        
        complexity_score = sum(
            factors[k] * weights[k]
            for k in factors
        )
        
        return {
            'score': complexity_score,
            'factors': factors,
            'query_type': self._determine_type(factors),
        }
    
    def _determine_type(self, factors: Dict[str, float]) -> QueryType:
        """Determine query type from factors."""
        if factors['word_count'] < 0.2 and factors['conjunctions'] == 0:
            return QueryType.SIMPLE
        elif factors['questions'] > 0.5:
            return QueryType.COMPOUND
        elif factors['conjunctions'] > 0.3 or factors['clauses'] > 0.5:
            return QueryType.COMPLEX
        return QueryType.NATURAL


class TemporalAnalyzer:
    """Analyze temporal references in queries."""
    
    TEMPORAL_PATTERNS = {
        TemporalReference.PAST: [
            r'\b(?:was|were|had|did|used\s+to)\b',
            r'\b(?:yesterday|last\s+\w+|ago|\d{4})\b',
            r'\b(?:previously|formerly|earlier|before)\b',
        ],
        TemporalReference.PRESENT: [
            r'\b(?:is|are|am|now|currently|today)\b',
            r'\b(?:at\s+present|these\s+days|nowadays)\b',
        ],
        TemporalReference.FUTURE: [
            r'\b(?:will|going\s+to|shall)\b',
            r'\b(?:tomorrow|next\s+\w+|soon|upcoming)\b',
            r'\b(?:in\s+the\s+future|later)\b',
        ],
    }
    
    def analyze(self, query: str) -> Tuple[TemporalReference, List[str]]:
        """
        Analyze temporal references.
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (temporal_reference, expressions)
        """
        expressions = []
        reference_counts = {ref: 0 for ref in TemporalReference}
        
        query_lower = query.lower()
        
        for ref, patterns in self.TEMPORAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                reference_counts[ref] += len(matches)
                expressions.extend(matches)
        
        # Determine primary temporal reference
        if not any(reference_counts.values()):
            return TemporalReference.NONE, []
        
        primary_ref = max(
            reference_counts,
            key=lambda r: reference_counts[r]
        )
        
        if reference_counts[primary_ref] == 0:
            return TemporalReference.NONE, []
        
        return primary_ref, list(set(expressions))


class QueryDecomposer:
    """Decompose complex queries into sub-queries."""
    
    def decompose(self, query: str) -> List[QueryComponent]:
        """
        Decompose query into components.
        
        Args:
            query: Input query
            
        Returns:
            List of query components
        """
        components = []
        
        # Split by conjunctions
        parts = re.split(
            r'\b(?:and|or|but|also|additionally)\b',
            query,
            flags=re.IGNORECASE
        )
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            # Clean up punctuation
            part = re.sub(r'^[,;:\s]+|[,;:\s]+$', '', part)
            
            if part:
                components.append(QueryComponent(
                    text=part,
                    type=self._classify_component(part),
                    importance=1.0 / (i + 1),  # Earlier = more important
                ))
        
        # Add dependencies
        self._add_dependencies(components)
        
        return components
    
    def generate_sub_queries(self, query: str) -> List[str]:
        """
        Generate sub-queries for complex queries.
        
        Args:
            query: Input query
            
        Returns:
            List of sub-queries
        """
        components = self.decompose(query)
        
        sub_queries = []
        for comp in components:
            # Ensure each sub-query is well-formed
            sub_query = comp.text
            if not sub_query.endswith('?'):
                if sub_query[0].isupper():
                    sub_query = sub_query + '?'
            sub_queries.append(sub_query)
        
        return sub_queries
    
    def _classify_component(self, text: str) -> str:
        """Classify component type."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['what', 'which', 'who']):
            return 'question'
        elif any(w in text_lower for w in ['how', 'why']):
            return 'explanation'
        elif any(w in text_lower for w in ['compare', 'vs', 'versus']):
            return 'comparison'
        return 'statement'
    
    def _add_dependencies(self, components: List[QueryComponent]) -> None:
        """Add dependencies between components."""
        for i, comp in enumerate(components):
            # Check for references to previous components
            if any(w in comp.text.lower() for w in ['it', 'this', 'that', 'they']):
                if i > 0:
                    comp.dependencies.append(i - 1)


class QueryAnalyzer:
    """
    Main query analysis interface.
    
    Example:
        >>> analyzer = QueryAnalyzer()
        >>> 
        >>> # Basic analysis
        >>> analysis = analyzer.analyze("What is machine learning?")
        >>> print(f"Intent: {analysis.intent}")
        >>> print(f"Entities: {analysis.entities}")
        >>> 
        >>> # Complex query
        >>> analysis = analyzer.analyze(
        ...     "Compare Python and JavaScript for web development"
        ... )
        >>> print(f"Type: {analysis.query_type}")
        >>> print(f"Sub-queries: {analysis.sub_queries}")
    """
    
    def __init__(
        self,
        enable_decomposition: bool = True,
        enable_temporal: bool = True,
        min_complexity_for_decomposition: float = 0.5,
    ):
        """
        Initialize query analyzer.
        
        Args:
            enable_decomposition: Enable query decomposition
            enable_temporal: Enable temporal analysis
            min_complexity_for_decomposition: Min complexity for decomposition
        """
        self.enable_decomposition = enable_decomposition
        self.enable_temporal = enable_temporal
        self.min_complexity_for_decomposition = min_complexity_for_decomposition
        
        # Components
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.keyword_extractor = KeywordExtractor()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.query_decomposer = QueryDecomposer()
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.
        
        Args:
            query: Input query
            
        Returns:
            QueryAnalysis
        """
        # Intent classification
        intent, intent_confidence = self.intent_classifier.classify(query)
        
        # Entity extraction
        entities = self.entity_extractor.extract(query)
        
        # Keyword extraction
        keywords, key_phrases = self.keyword_extractor.extract(query)
        
        # Complexity analysis
        complexity = self.complexity_analyzer.analyze(query)
        complexity_score = complexity['score']
        query_type = complexity['query_type']
        
        # Temporal analysis
        temporal_ref = TemporalReference.NONE
        temporal_exprs = []
        if self.enable_temporal:
            temporal_ref, temporal_exprs = self.temporal_analyzer.analyze(query)
        
        # Query decomposition
        components = []
        sub_queries = []
        if (self.enable_decomposition and 
            complexity_score >= self.min_complexity_for_decomposition):
            components = self.query_decomposer.decompose(query)
            sub_queries = self.query_decomposer.generate_sub_queries(query)
        
        # Quality assessment
        is_well_formed = self._assess_well_formed(query)
        clarity_score = self._assess_clarity(query)
        specificity_score = self._assess_specificity(query, entities, keywords)
        
        # Estimate context length needed
        estimated_context = self._estimate_context_length(
            complexity_score, len(entities), len(sub_queries)
        )
        
        return QueryAnalysis(
            query=query,
            intent=intent,
            intent_confidence=intent_confidence,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            key_phrases=key_phrases,
            components=components,
            temporal_reference=temporal_ref,
            temporal_expressions=temporal_exprs,
            complexity_score=complexity_score,
            estimated_context_length=estimated_context,
            is_well_formed=is_well_formed,
            clarity_score=clarity_score,
            specificity_score=specificity_score,
            sub_queries=sub_queries,
        )
    
    def _assess_well_formed(self, query: str) -> bool:
        """Assess if query is well-formed."""
        # Check basic structure
        if len(query.strip()) < 3:
            return False
        
        # Check for complete words
        words = query.split()
        if len(words) < 1:
            return False
        
        # Check for proper ending
        if query.strip()[-1] not in '.?!':
            return True  # Still valid, just no punctuation
        
        return True
    
    def _assess_clarity(self, query: str) -> float:
        """Assess query clarity."""
        score = 1.0
        
        # Penalize very short queries
        if len(query.split()) < 3:
            score *= 0.7
        
        # Penalize all caps
        if query.isupper():
            score *= 0.8
        
        # Penalize excessive punctuation
        punct_count = sum(1 for c in query if c in '!?.,;:')
        if punct_count > len(query) * 0.1:
            score *= 0.9
        
        return score
    
    def _assess_specificity(
        self,
        query: str,
        entities: List[Entity],
        keywords: List[str],
    ) -> float:
        """Assess query specificity."""
        score = 0.5  # Base score
        
        # More entities = more specific
        score += min(len(entities) * 0.1, 0.3)
        
        # More keywords = more specific
        score += min(len(keywords) * 0.05, 0.2)
        
        # Named entities are very specific
        named_entities = [e for e in entities if e.type in {
            EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION
        }]
        score += min(len(named_entities) * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _estimate_context_length(
        self,
        complexity: float,
        entity_count: int,
        sub_query_count: int,
    ) -> int:
        """Estimate required context length."""
        base = 1
        
        # More complex = more context
        base += int(complexity * 3)
        
        # More entities = more context
        base += entity_count
        
        # More sub-queries = more context
        base += sub_query_count
        
        return min(base, 10)


# Convenience functions

def analyze_query(query: str) -> QueryAnalysis:
    """
    Quick query analysis.
    
    Args:
        query: Input query
        
    Returns:
        QueryAnalysis
    """
    analyzer = QueryAnalyzer()
    return analyzer.analyze(query)


def get_query_intent(query: str) -> QueryIntent:
    """
    Get query intent.
    
    Args:
        query: Input query
        
    Returns:
        QueryIntent
    """
    classifier = IntentClassifier()
    intent, _ = classifier.classify(query)
    return intent


def extract_entities(query: str) -> List[Entity]:
    """
    Extract entities from query.
    
    Args:
        query: Input query
        
    Returns:
        List of entities
    """
    extractor = EntityExtractor()
    return extractor.extract(query)


def decompose_query(query: str) -> List[str]:
    """
    Decompose complex query into sub-queries.
    
    Args:
        query: Input query
        
    Returns:
        List of sub-queries
    """
    decomposer = QueryDecomposer()
    return decomposer.generate_sub_queries(query)
