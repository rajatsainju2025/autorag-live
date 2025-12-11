"""Query Understanding Module for AutoRAG-Live.

Parse and analyze queries to extract:
- Intent classification
- Entity extraction
- Query type detection
- Temporal expressions
- Query complexity analysis
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents."""
    
    FACTUAL = "factual"  # Looking for facts
    DEFINITIONAL = "definitional"  # What is X?
    PROCEDURAL = "procedural"  # How to do X?
    COMPARATIVE = "comparative"  # Compare X and Y
    CAUSAL = "causal"  # Why/what caused X?
    OPINION = "opinion"  # What do people think about X?
    NAVIGATIONAL = "navigational"  # Find specific resource
    TRANSACTIONAL = "transactional"  # Perform action
    EXPLORATORY = "exploratory"  # General exploration
    UNKNOWN = "unknown"


class QueryType(Enum):
    """Types of queries."""
    
    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"
    KEYWORD = "keyword"


class EntityType(Enum):
    """Types of named entities."""
    
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    NUMBER = "number"
    PRODUCT = "product"
    EVENT = "event"
    CONCEPT = "concept"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Extracted entity from query."""
    
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalExpression:
    """Temporal expression in query."""
    
    text: str
    temporal_type: str  # absolute, relative, range
    start: int
    end: int
    normalized: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryComplexity:
    """Query complexity analysis."""
    
    word_count: int
    token_count: int
    entity_count: int
    clause_count: int
    complexity_score: float  # 0-1 scale
    is_compound: bool
    sub_queries: list[str] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    
    original_query: str
    normalized_query: str
    query_type: QueryType
    primary_intent: QueryIntent
    secondary_intents: list[QueryIntent]
    entities: list[Entity]
    temporal_expressions: list[TemporalExpression]
    complexity: QueryComplexity
    keywords: list[str]
    question_words: list[str]
    confidence: float
    analysis_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """Abstract base class for query analyzers."""
    
    @abstractmethod
    def analyze(self, query: str) -> dict[str, Any]:
        """Analyze query and return results.
        
        Args:
            query: Query string to analyze
            
        Returns:
            Analysis results
        """
        pass


class IntentClassifier(BaseAnalyzer):
    """Classify query intent."""
    
    # Intent patterns
    INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
        QueryIntent.DEFINITIONAL: [
            r"^what\s+is\b",
            r"^what\s+are\b",
            r"^define\b",
            r"^meaning\s+of\b",
            r"^definition\s+of\b",
        ],
        QueryIntent.PROCEDURAL: [
            r"^how\s+to\b",
            r"^how\s+do\s+(i|you|we)\b",
            r"^how\s+can\s+(i|you|we)\b",
            r"^steps\s+to\b",
            r"^guide\s+to\b",
            r"^tutorial\b",
        ],
        QueryIntent.COMPARATIVE: [
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bdifference\s+between\b",
            r"\bbetter\s+than\b",
            r"\bworse\s+than\b",
        ],
        QueryIntent.CAUSAL: [
            r"^why\b",
            r"\bcause\b",
            r"\breason\b",
            r"\bbecause\b",
            r"\bresult\s+in\b",
            r"\blead\s+to\b",
        ],
        QueryIntent.OPINION: [
            r"\bopinion\b",
            r"\bthink\s+about\b",
            r"\bfeel\s+about\b",
            r"\breview\b",
            r"\brecommend\b",
            r"\bbest\b",
            r"\bworst\b",
        ],
        QueryIntent.NAVIGATIONAL: [
            r"\bwebsite\b",
            r"\blink\b",
            r"\burl\b",
            r"\bfind\s+.*\s+page\b",
            r"\bofficial\b",
            r"\bhomepage\b",
        ],
        QueryIntent.TRANSACTIONAL: [
            r"^buy\b",
            r"^purchase\b",
            r"^download\b",
            r"^install\b",
            r"^subscribe\b",
            r"^sign\s+up\b",
            r"^register\b",
        ],
        QueryIntent.FACTUAL: [
            r"^when\b",
            r"^where\b",
            r"^who\b",
            r"^which\b",
            r"^how\s+many\b",
            r"^how\s+much\b",
            r"\bfact\b",
        ],
    }
    
    def analyze(self, query: str) -> dict[str, Any]:
        """Classify query intent."""
        query_lower = query.lower().strip()
        
        intent_scores: dict[QueryIntent, float] = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
            intent_scores[intent] = score
        
        # Sort by score
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Primary intent
        primary_intent = QueryIntent.UNKNOWN
        secondary_intents: list[QueryIntent] = []
        
        if sorted_intents and sorted_intents[0][1] > 0:
            primary_intent = sorted_intents[0][0]
            secondary_intents = [
                intent for intent, score in sorted_intents[1:] if score > 0
            ]
        
        # Default to exploratory if no patterns match
        if primary_intent == QueryIntent.UNKNOWN:
            primary_intent = QueryIntent.EXPLORATORY
        
        confidence = min(1.0, max(intent_scores.values()) / 2) if intent_scores else 0.5
        
        return {
            "primary_intent": primary_intent,
            "secondary_intents": secondary_intents,
            "intent_scores": intent_scores,
            "confidence": confidence,
        }


class EntityExtractor(BaseAnalyzer):
    """Extract named entities from query."""
    
    # Simple pattern-based entity extraction
    ENTITY_PATTERNS: dict[EntityType, list[tuple[str, str]]] = {
        EntityType.DATE: [
            (r"\b\d{4}-\d{2}-\d{2}\b", "iso_date"),
            (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "slash_date"),
            (r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s*\d{4}\b", "written_date"),
            (r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b", "euro_date"),
        ],
        EntityType.TIME: [
            (r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", "time"),
        ],
        EntityType.NUMBER: [
            (r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b", "currency"),
            (r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|k|m|b)?\b", "number"),
            (r"\b\d+%\b", "percentage"),
        ],
        EntityType.LOCATION: [
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b", "city_state"),
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b", "city_country"),
        ],
    }
    
    # Common organization indicators
    ORG_SUFFIXES = [
        "inc", "corp", "llc", "ltd", "co", "company", "corporation",
        "foundation", "institute", "university", "college",
    ]
    
    def analyze(self, query: str) -> dict[str, Any]:
        """Extract entities from query."""
        entities: list[Entity] = []
        
        # Pattern-based extraction
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern, pattern_name in patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    entities.append(Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        metadata={"pattern": pattern_name},
                    ))
        
        # Organization detection
        for suffix in self.ORG_SUFFIXES:
            pattern = rf"\b[A-Z][a-zA-Z\s]+\s+{suffix}\b"
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    entity_type=EntityType.ORGANIZATION,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    metadata={"suffix": suffix},
                ))
        
        # Capitalized phrases (potential entities)
        for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query):
            # Check if not already captured
            text = match.group()
            already_captured = any(
                e.start <= match.start() < e.end or e.start < match.end() <= e.end
                for e in entities
            )
            if not already_captured:
                entities.append(Entity(
                    text=text,
                    entity_type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.5,
                    metadata={"detection": "capitalized_phrase"},
                ))
        
        # Remove duplicates
        seen_spans = set()
        unique_entities = []
        for entity in entities:
            span = (entity.start, entity.end)
            if span not in seen_spans:
                seen_spans.add(span)
                unique_entities.append(entity)
        
        return {"entities": unique_entities}


class TemporalExtractor(BaseAnalyzer):
    """Extract temporal expressions from query."""
    
    TEMPORAL_PATTERNS = [
        # Relative
        (r"\b(today|yesterday|tomorrow)\b", "relative"),
        (r"\b(this|last|next)\s+(week|month|year|quarter)\b", "relative"),
        (r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+(ago|from\s+now)\b", "relative"),
        (r"\b(recently|lately|soon|earlier|later)\b", "relative"),
        
        # Absolute
        (r"\b\d{4}\b", "absolute"),
        (r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b", "absolute"),
        
        # Range
        (r"\b(from|between)\s+.+\s+(to|and)\s+.+\b", "range"),
        (r"\b(since|until|before|after)\s+\d{4}\b", "range"),
    ]
    
    def analyze(self, query: str) -> dict[str, Any]:
        """Extract temporal expressions."""
        temporal_expressions: list[TemporalExpression] = []
        
        for pattern, temporal_type in self.TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                temporal_expressions.append(TemporalExpression(
                    text=match.group(),
                    temporal_type=temporal_type,
                    start=match.start(),
                    end=match.end(),
                    metadata={"pattern": pattern},
                ))
        
        return {"temporal_expressions": temporal_expressions}


class QueryTypeDetector(BaseAnalyzer):
    """Detect query type (question, statement, command, keyword)."""
    
    QUESTION_PATTERNS = [
        r"^(what|who|where|when|why|how|which|whose|whom)\b",
        r"\?$",
        r"^(is|are|was|were|do|does|did|can|could|will|would|should)\b",
    ]
    
    COMMAND_PATTERNS = [
        r"^(show|find|get|list|search|display|give|tell|explain)\b",
        r"^(create|make|build|generate|produce)\b",
        r"^(please\s+)?(show|find|get|list|search)\b",
    ]
    
    def analyze(self, query: str) -> dict[str, Any]:
        """Detect query type."""
        query_stripped = query.strip()
        query_lower = query_stripped.lower()
        
        # Check for question
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, query_lower):
                return {"query_type": QueryType.QUESTION}
        
        # Check for command
        for pattern in self.COMMAND_PATTERNS:
            if re.search(pattern, query_lower):
                return {"query_type": QueryType.COMMAND}
        
        # Check for keyword (no spaces, short)
        if " " not in query_stripped and len(query_stripped) < 30:
            return {"query_type": QueryType.KEYWORD}
        
        # Default to statement
        return {"query_type": QueryType.STATEMENT}


class ComplexityAnalyzer(BaseAnalyzer):
    """Analyze query complexity."""
    
    CLAUSE_PATTERNS = [
        r"\b(and|or|but|however|although|because|since|while|if|when|where)\b",
    ]
    
    def analyze(self, query: str) -> dict[str, Any]:
        """Analyze query complexity."""
        words = query.split()
        word_count = len(words)
        
        # Estimate token count (rough approximation)
        token_count = int(word_count * 1.3)
        
        # Count clauses
        clause_indicators = 0
        for pattern in self.CLAUSE_PATTERNS:
            clause_indicators += len(re.findall(pattern, query, re.IGNORECASE))
        
        clause_count = max(1, clause_indicators + 1)
        
        # Detect compound queries
        is_compound = clause_count > 1 or "?" in query[:-1]
        
        # Extract sub-queries if compound
        sub_queries: list[str] = []
        if is_compound:
            # Split by conjunctions
            parts = re.split(r"\s+(?:and|or)\s+", query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if len(p.strip()) > 3]
        
        # Complexity score (0-1)
        complexity_score = min(1.0, (
            word_count / 30 * 0.3 +
            clause_count / 3 * 0.3 +
            (1 if is_compound else 0) * 0.2 +
            len(sub_queries) / 3 * 0.2
        ))
        
        return {
            "complexity": QueryComplexity(
                word_count=word_count,
                token_count=token_count,
                entity_count=0,  # Updated later
                clause_count=clause_count,
                complexity_score=complexity_score,
                is_compound=is_compound,
                sub_queries=sub_queries,
            )
        }


class KeywordExtractor(BaseAnalyzer):
    """Extract keywords from query."""
    
    # Common stop words
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "up",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "also",
        "now", "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
        "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "and", "but", "if", "or", "because", "as", "until", "while",
    }
    
    # Question words to preserve
    QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which"}
    
    def analyze(self, query: str) -> dict[str, Any]:
        """Extract keywords from query."""
        # Tokenize
        words = re.findall(r"\b\w+\b", query.lower())
        
        # Extract question words
        question_words = [w for w in words if w in self.QUESTION_WORDS]
        
        # Extract keywords (excluding stop words)
        keywords = [
            w for w in words
            if w not in self.STOP_WORDS and len(w) > 2
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return {
            "keywords": unique_keywords,
            "question_words": question_words,
        }


class QueryNormalizer:
    """Normalize queries for consistent processing."""
    
    def normalize(self, query: str) -> str:
        """Normalize query string.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query
        """
        # Strip whitespace
        normalized = query.strip()
        
        # Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized)
        
        # Standardize quotes
        normalized = normalized.replace("'", "'").replace(""", '"').replace(""", '"')
        
        # Remove excessive punctuation
        normalized = re.sub(r"[!?]{2,}", "?", normalized)
        
        # Fix common typos/patterns
        normalized = re.sub(r"\bwanna\b", "want to", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bgonna\b", "going to", normalized, flags=re.IGNORECASE)
        
        return normalized


class QueryUnderstandingEngine:
    """Main engine for query understanding."""
    
    def __init__(
        self,
        custom_analyzers: list[BaseAnalyzer] | None = None,
    ) -> None:
        """Initialize query understanding engine.
        
        Args:
            custom_analyzers: Additional custom analyzers
        """
        self.normalizer = QueryNormalizer()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.temporal_extractor = TemporalExtractor()
        self.query_type_detector = QueryTypeDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        
        self.custom_analyzers = custom_analyzers or []
        
        # Statistics
        self._stats: dict[str, Any] = {
            "total_queries": 0,
            "total_analysis_time_ms": 0.0,
            "intent_distribution": {},
        }
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query comprehensively.
        
        Args:
            query: Query string to analyze
            
        Returns:
            Complete query analysis
        """
        start_time = time.time()
        
        # Normalize query
        normalized = self.normalizer.normalize(query)
        
        # Run all analyzers
        intent_result = self.intent_classifier.analyze(normalized)
        entity_result = self.entity_extractor.analyze(normalized)
        temporal_result = self.temporal_extractor.analyze(normalized)
        type_result = self.query_type_detector.analyze(normalized)
        complexity_result = self.complexity_analyzer.analyze(normalized)
        keyword_result = self.keyword_extractor.analyze(normalized)
        
        # Update entity count in complexity
        complexity = complexity_result["complexity"]
        complexity.entity_count = len(entity_result["entities"])
        
        # Run custom analyzers
        custom_results: dict[str, Any] = {}
        for analyzer in self.custom_analyzers:
            try:
                result = analyzer.analyze(normalized)
                custom_results.update(result)
            except Exception as e:
                logger.warning(f"Custom analyzer failed: {e}")
        
        # Calculate confidence
        confidence = (
            intent_result["confidence"] * 0.4 +
            (0.8 if entity_result["entities"] else 0.5) * 0.3 +
            (1.0 - complexity.complexity_score * 0.5) * 0.3
        )
        
        analysis_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self._stats["total_queries"] += 1
        self._stats["total_analysis_time_ms"] += analysis_time_ms
        
        intent_name = intent_result["primary_intent"].value
        self._stats["intent_distribution"][intent_name] = (
            self._stats["intent_distribution"].get(intent_name, 0) + 1
        )
        
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            query_type=type_result["query_type"],
            primary_intent=intent_result["primary_intent"],
            secondary_intents=intent_result["secondary_intents"],
            entities=entity_result["entities"],
            temporal_expressions=temporal_result["temporal_expressions"],
            complexity=complexity,
            keywords=keyword_result["keywords"],
            question_words=keyword_result["question_words"],
            confidence=confidence,
            analysis_time_ms=analysis_time_ms,
            metadata=custom_results,
        )
    
    def get_search_query(self, analysis: QueryAnalysis) -> str:
        """Generate optimized search query from analysis.
        
        Args:
            analysis: Query analysis result
            
        Returns:
            Optimized search query
        """
        # Start with keywords
        search_terms = analysis.keywords.copy()
        
        # Add entity text
        for entity in analysis.entities:
            if entity.text.lower() not in [k.lower() for k in search_terms]:
                search_terms.append(entity.text)
        
        # Build search query
        search_query = " ".join(search_terms[:10])  # Limit terms
        
        return search_query
    
    def get_query_filters(
        self,
        analysis: QueryAnalysis,
    ) -> dict[str, Any]:
        """Extract filters from query analysis.
        
        Args:
            analysis: Query analysis result
            
        Returns:
            Filter dictionary
        """
        filters: dict[str, Any] = {}
        
        # Extract temporal filters
        if analysis.temporal_expressions:
            filters["temporal"] = [
                {
                    "text": te.text,
                    "type": te.temporal_type,
                    "normalized": te.normalized,
                }
                for te in analysis.temporal_expressions
            ]
        
        # Extract entity filters
        for entity in analysis.entities:
            entity_type = entity.entity_type.value
            if entity_type not in filters:
                filters[entity_type] = []
            filters[entity_type].append(entity.text)
        
        return filters
    
    def get_stats(self) -> dict[str, Any]:
        """Get query understanding statistics."""
        stats = self._stats.copy()
        
        if stats["total_queries"] > 0:
            stats["avg_analysis_time_ms"] = (
                stats["total_analysis_time_ms"] / stats["total_queries"]
            )
        else:
            stats["avg_analysis_time_ms"] = 0.0
        
        return stats


# Convenience function
def analyze_query(query: str) -> QueryAnalysis:
    """Convenience function to analyze a query.
    
    Args:
        query: Query string
        
    Returns:
        Query analysis
    """
    engine = QueryUnderstandingEngine()
    return engine.analyze(query)
