"""
Query intent classification for AutoRAG-Live.

Classifies user queries into intent categories to enable
intelligent routing and response customization.

Intent categories:
- FACTUAL: Direct fact-seeking questions
- ANALYTICAL: Questions requiring analysis/reasoning
- COMPARISON: Comparing multiple entities
- OPINION: Seeking opinions or recommendations
- INSTRUCTION: How-to questions
- DEFINITION: Definition/explanation requests
- TEMPORAL: Time-related questions
- SPATIAL: Location-related questions
- CAUSAL: Why/cause-effect questions
- LISTING: List-type questions

Example usage:
    >>> classifier = IntentClassifier()
    >>> result = classifier.classify("What is the capital of France?")
    >>> print(result.intent)  # IntentType.FACTUAL

    >>> # Multi-label classification
    >>> result = classifier.classify("Compare Python vs JavaScript for web dev")
    >>> print(result.intents)  # [IntentType.COMPARISON, IntentType.ANALYTICAL]

    >>> # Async embedding-based classification
    >>> async_classifier = AsyncIntentClassifier(embedder)
    >>> result = await async_classifier.classify("How do I install Python?")
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class EmbedderProtocol(Protocol):
    """Protocol for embedder interface."""

    async def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        ...


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


class IntentType(str, Enum):
    """Query intent categories."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    OPINION = "opinion"
    INSTRUCTION = "instruction"
    DEFINITION = "definition"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    LISTING = "listing"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    """High-level query types."""

    QUESTION = "question"
    COMMAND = "command"
    STATEMENT = "statement"
    INCOMPLETE = "incomplete"


@dataclass
class IntentFeatures:
    """Extracted features from query for classification."""

    query: str
    tokens: List[str] = field(default_factory=list)
    question_word: Optional[str] = None
    has_question_mark: bool = False
    word_count: int = 0
    entities: List[str] = field(default_factory=list)
    comparison_terms: List[str] = field(default_factory=list)
    temporal_terms: List[str] = field(default_factory=list)
    listing_patterns: bool = False
    instruction_patterns: bool = False


@dataclass
class IntentResult:
    """Result of intent classification."""

    query: str
    intent: IntentType
    confidence: float
    query_type: QueryType

    # Multi-label support
    intents: List[Tuple[IntentType, float]] = field(default_factory=list)

    # Features used
    features: Optional[IntentFeatures] = None

    # Metadata
    processing_time_ms: float = 0.0
    classifier_version: str = "1.0"

    @property
    def top_intents(self) -> List[IntentType]:
        """Get top intents above threshold."""
        return [intent for intent, score in self.intents if score >= 0.5]

    def has_intent(self, intent: IntentType, threshold: float = 0.5) -> bool:
        """Check if query has specific intent."""
        for i, score in self.intents:
            if i == intent and score >= threshold:
                return True
        return False


class FeatureExtractor:
    """Extract features from queries for classification."""

    # Question words and their typical intents
    QUESTION_WORDS = {
        "what": [IntentType.FACTUAL, IntentType.DEFINITION],
        "who": [IntentType.FACTUAL],
        "when": [IntentType.TEMPORAL, IntentType.FACTUAL],
        "where": [IntentType.SPATIAL, IntentType.FACTUAL],
        "why": [IntentType.CAUSAL, IntentType.ANALYTICAL],
        "how": [IntentType.INSTRUCTION, IntentType.ANALYTICAL],
        "which": [IntentType.FACTUAL, IntentType.COMPARISON],
        "is": [IntentType.FACTUAL],
        "are": [IntentType.FACTUAL],
        "do": [IntentType.FACTUAL],
        "does": [IntentType.FACTUAL],
        "can": [IntentType.FACTUAL, IntentType.INSTRUCTION],
        "could": [IntentType.OPINION, IntentType.INSTRUCTION],
        "should": [IntentType.OPINION, IntentType.ANALYTICAL],
        "would": [IntentType.OPINION, IntentType.ANALYTICAL],
    }

    # Comparison indicators
    COMPARISON_TERMS = {
        "vs",
        "versus",
        "compared",
        "compare",
        "comparing",
        "comparison",
        "difference",
        "differences",
        "different",
        "differ",
        "better",
        "worse",
        "best",
        "worst",
        "similar",
        "similarity",
        "similarities",
        "prefer",
        "preference",
        "alternative",
        "alternatives",
        "or",
        "rather",
    }

    # Temporal indicators
    TEMPORAL_TERMS = {
        "when",
        "time",
        "date",
        "year",
        "month",
        "day",
        "week",
        "before",
        "after",
        "during",
        "since",
        "until",
        "today",
        "tomorrow",
        "yesterday",
        "recent",
        "recently",
        "latest",
        "newest",
        "history",
        "historical",
        "past",
        "future",
        "now",
        "current",
        "currently",
        "first",
        "last",
        "next",
        "previous",
    }

    # Spatial indicators
    SPATIAL_TERMS = {
        "where",
        "location",
        "place",
        "address",
        "city",
        "country",
        "near",
        "nearby",
        "closest",
        "distance",
        "region",
        "area",
        "zone",
        "territory",
        "north",
        "south",
        "east",
        "west",
        "here",
        "there",
        "somewhere",
    }

    # Listing indicators
    LISTING_PATTERNS = [
        r"\blist\b",
        r"\bshow\s+(?:me\s+)?(?:all|the|some)\b",
        r"\bwhat\s+(?:are|is)\s+(?:all|the|some)\b",
        r"\bgive\s+(?:me\s+)?(?:all|the|some)\b",
        r"\btop\s+\d+\b",
        r"\bbest\s+\d+\b",
        r"\bexamples?\s+of\b",
        r"\btypes?\s+of\b",
        r"\bkinds?\s+of\b",
    ]

    # Instruction indicators
    INSTRUCTION_PATTERNS = [
        r"\bhow\s+(?:to|do|can|should)\b",
        r"\bsteps?\s+(?:to|for)\b",
        r"\bguide\s+(?:to|for|on)\b",
        r"\btutorial\b",
        r"\binstruction(?:s)?\b",
        r"\bprocess\s+(?:to|for|of)\b",
        r"\bcreate\b",
        r"\bbuild\b",
        r"\bmake\b",
        r"\bsetup\b",
        r"\binstall\b",
        r"\bconfigure\b",
    ]

    # Definition indicators
    DEFINITION_PATTERNS = [
        r"\bwhat\s+is\b",
        r"\bwhat\s+are\b",
        r"\bdefine\b",
        r"\bdefinition\s+of\b",
        r"\bmeaning\s+of\b",
        r"\bexplain\b",
        r"\bdescribe\b",
    ]

    # Opinion indicators
    OPINION_PATTERNS = [
        r"\bshould\s+i\b",
        r"\bwould\s+you\b",
        r"\brecommend\b",
        r"\bsuggestion(?:s)?\b",
        r"\badvice\b",
        r"\bopinion(?:s)?\b",
        r"\bthink\s+(?:about|of)\b",
        r"\bbest\s+(?:way|approach|method)\b",
        r"\bworth\s+(?:it)?\b",
    ]

    def __init__(self):
        """Initialize feature extractor."""
        # Compile patterns
        self._listing_patterns = [re.compile(p, re.IGNORECASE) for p in self.LISTING_PATTERNS]
        self._instruction_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INSTRUCTION_PATTERNS
        ]
        self._definition_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEFINITION_PATTERNS]
        self._opinion_patterns = [re.compile(p, re.IGNORECASE) for p in self.OPINION_PATTERNS]

    def extract(self, query: str) -> IntentFeatures:
        """
        Extract features from query.

        Args:
            query: Input query string

        Returns:
            IntentFeatures with extracted features
        """
        # Normalize
        normalized = query.strip().lower()
        tokens = normalized.split()

        features = IntentFeatures(
            query=query,
            tokens=tokens,
            word_count=len(tokens),
            has_question_mark=query.strip().endswith("?"),
        )

        # Extract question word
        if tokens:
            first_word = tokens[0]
            if first_word in self.QUESTION_WORDS:
                features.question_word = first_word

        # Find comparison terms
        features.comparison_terms = [t for t in tokens if t in self.COMPARISON_TERMS]

        # Find temporal terms
        features.temporal_terms = [t for t in tokens if t in self.TEMPORAL_TERMS]

        # Check listing patterns
        features.listing_patterns = any(p.search(normalized) for p in self._listing_patterns)

        # Check instruction patterns
        features.instruction_patterns = any(
            p.search(normalized) for p in self._instruction_patterns
        )

        return features


class RuleBasedClassifier:
    """Rule-based intent classifier using heuristics."""

    def __init__(self):
        """Initialize classifier."""
        self.extractor = FeatureExtractor()

    def classify(self, query: str) -> Dict[IntentType, float]:
        """
        Classify query intent using rules.

        Args:
            query: Input query

        Returns:
            Dict mapping intents to confidence scores
        """
        features = self.extractor.extract(query)
        normalized = query.strip().lower()

        scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}

        # Question word signals
        if features.question_word:
            intents = self.extractor.QUESTION_WORDS.get(features.question_word, [])
            for intent in intents:
                scores[intent] += 0.3

        # Question mark boost
        if features.has_question_mark:
            scores[IntentType.FACTUAL] += 0.1

        # Comparison detection
        if features.comparison_terms:
            scores[IntentType.COMPARISON] += 0.4 + 0.1 * len(features.comparison_terms)
            if "or" in features.comparison_terms:
                scores[IntentType.COMPARISON] += 0.2

        # Temporal detection
        if features.temporal_terms:
            scores[IntentType.TEMPORAL] += 0.3 + 0.1 * len(features.temporal_terms)

        # Spatial detection
        spatial_terms = [t for t in features.tokens if t in self.extractor.SPATIAL_TERMS]
        if spatial_terms:
            scores[IntentType.SPATIAL] += 0.3 + 0.1 * len(spatial_terms)

        # Listing detection
        if features.listing_patterns:
            scores[IntentType.LISTING] += 0.5

        # Instruction detection
        if features.instruction_patterns:
            scores[IntentType.INSTRUCTION] += 0.5

        # Definition patterns
        for pattern in self.extractor._definition_patterns:
            if pattern.search(normalized):
                scores[IntentType.DEFINITION] += 0.4
                break

        # Opinion patterns
        for pattern in self.extractor._opinion_patterns:
            if pattern.search(normalized):
                scores[IntentType.OPINION] += 0.4
                break

        # Causal patterns
        causal_patterns = [
            r"\bwhy\b",
            r"\bcause(?:s|d)?\b",
            r"\breason(?:s)?\b",
            r"\bbecause\b",
            r"\bdue\s+to\b",
            r"\bresult(?:s|ed)?\s+(?:in|from)\b",
        ]
        for pattern in causal_patterns:
            if re.search(pattern, normalized):
                scores[IntentType.CAUSAL] += 0.35
                break

        # Analytical patterns
        analytical_patterns = [
            r"\banalyze\b",
            r"\banalysis\b",
            r"\bevaluate\b",
            r"\bassess\b",
            r"\bimpact\b",
            r"\beffect(?:s)?\b",
            r"\bimplication(?:s)?\b",
            r"\bpro(?:s)?\s+and\s+con(?:s)?\b",
        ]
        for pattern in analytical_patterns:
            if re.search(pattern, normalized):
                scores[IntentType.ANALYTICAL] += 0.4
                break

        # Conversational patterns
        conversational_patterns = [
            r"^(?:hi|hello|hey)\b",
            r"^(?:thanks?|thank\s+you)\b",
            r"^(?:goodbye|bye)\b",
            r"^(?:please|plz)\b",
            r"\b(?:how\s+are\s+you)\b",
        ]
        for pattern in conversational_patterns:
            if re.search(pattern, normalized):
                scores[IntentType.CONVERSATIONAL] += 0.6
                break

        # Normalize scores
        max_score = max(scores.values()) if scores else 0.0
        if max_score > 0:
            for intent in scores:
                scores[intent] = min(
                    1.0, scores[intent] / max_score if max_score > 1.0 else scores[intent]
                )

        # Ensure at least factual if no strong signal
        if max(scores.values()) < 0.3 and features.has_question_mark:
            scores[IntentType.FACTUAL] = 0.4
        elif max(scores.values()) < 0.2:
            scores[IntentType.UNKNOWN] = 0.5

        return scores


class KeywordClassifier:
    """Keyword-based intent classifier with weighted terms."""

    # Intent keyword weights
    INTENT_KEYWORDS: Dict[IntentType, Dict[str, float]] = {
        IntentType.FACTUAL: {
            "what": 0.3,
            "who": 0.4,
            "where": 0.3,
            "when": 0.3,
            "is": 0.2,
            "are": 0.2,
            "was": 0.2,
            "were": 0.2,
            "name": 0.3,
            "number": 0.3,
            "count": 0.3,
            "fact": 0.4,
            "true": 0.3,
            "false": 0.3,
        },
        IntentType.ANALYTICAL: {
            "analyze": 0.5,
            "analysis": 0.5,
            "evaluate": 0.5,
            "assess": 0.4,
            "examine": 0.4,
            "investigate": 0.4,
            "why": 0.3,
            "impact": 0.4,
            "effect": 0.4,
            "trend": 0.4,
            "pattern": 0.4,
            "correlation": 0.5,
        },
        IntentType.COMPARISON: {
            "vs": 0.6,
            "versus": 0.6,
            "compare": 0.6,
            "comparison": 0.6,
            "difference": 0.5,
            "differences": 0.5,
            "different": 0.4,
            "similar": 0.4,
            "similarity": 0.5,
            "better": 0.4,
            "worse": 0.4,
            "prefer": 0.4,
            "alternative": 0.4,
            "or": 0.2,
        },
        IntentType.OPINION: {
            "should": 0.4,
            "recommend": 0.5,
            "suggestion": 0.5,
            "advice": 0.5,
            "opinion": 0.5,
            "think": 0.3,
            "best": 0.3,
            "prefer": 0.4,
            "worth": 0.4,
            "favorite": 0.4,
            "ideal": 0.4,
        },
        IntentType.INSTRUCTION: {
            "how": 0.4,
            "steps": 0.5,
            "step": 0.5,
            "guide": 0.5,
            "tutorial": 0.5,
            "instructions": 0.5,
            "process": 0.4,
            "create": 0.4,
            "build": 0.4,
            "make": 0.3,
            "setup": 0.5,
            "install": 0.5,
            "configure": 0.5,
            "implement": 0.4,
        },
        IntentType.DEFINITION: {
            "define": 0.6,
            "definition": 0.6,
            "meaning": 0.5,
            "explain": 0.4,
            "describe": 0.4,
            "what": 0.2,
            "concept": 0.4,
            "term": 0.4,
            "terminology": 0.5,
        },
        IntentType.TEMPORAL: {
            "when": 0.5,
            "time": 0.4,
            "date": 0.5,
            "year": 0.4,
            "month": 0.4,
            "day": 0.4,
            "history": 0.4,
            "before": 0.3,
            "after": 0.3,
            "during": 0.3,
            "recent": 0.4,
            "latest": 0.4,
            "current": 0.3,
        },
        IntentType.SPATIAL: {
            "where": 0.5,
            "location": 0.5,
            "place": 0.4,
            "address": 0.5,
            "city": 0.4,
            "country": 0.4,
            "near": 0.4,
            "distance": 0.4,
            "region": 0.4,
        },
        IntentType.CAUSAL: {
            "why": 0.5,
            "cause": 0.5,
            "reason": 0.5,
            "because": 0.4,
            "result": 0.4,
            "lead": 0.3,
            "effect": 0.4,
            "consequence": 0.5,
            "outcome": 0.4,
        },
        IntentType.LISTING: {
            "list": 0.6,
            "show": 0.3,
            "all": 0.3,
            "examples": 0.5,
            "types": 0.4,
            "kinds": 0.4,
            "categories": 0.4,
            "top": 0.4,
            "best": 0.3,
            "options": 0.4,
        },
    }

    def classify(self, query: str) -> Dict[IntentType, float]:
        """
        Classify using keyword matching.

        Args:
            query: Input query

        Returns:
            Dict mapping intents to scores
        """
        tokens = query.lower().split()
        scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}

        for token in tokens:
            for intent, keywords in self.INTENT_KEYWORDS.items():
                if token in keywords:
                    scores[intent] += keywords[token]

        # Normalize
        max_score = max(scores.values()) if scores else 0.0
        if max_score > 1.0:
            for intent in scores:
                scores[intent] /= max_score

        return scores


class PatternClassifier:
    """Pattern-based classifier using regex matching."""

    # Intent patterns with confidence weights
    INTENT_PATTERNS: Dict[IntentType, List[Tuple[str, float]]] = {
        IntentType.FACTUAL: [
            (r"^what\s+is\s+(?:the\s+)?(?:\w+\s+){0,3}of\b", 0.6),
            (r"^who\s+(?:is|was|are|were)\b", 0.7),
            (r"^when\s+(?:did|was|is|will)\b", 0.7),
            (r"^where\s+(?:is|are|was|were|can)\b", 0.7),
            (r"\bhow\s+(?:much|many|old|long|far)\b", 0.6),
            (r"^is\s+(?:it|there|this)\b", 0.5),
            (r"^does\s+\w+\s+\w+\b", 0.5),
        ],
        IntentType.ANALYTICAL: [
            (r"\banalyze\s+(?:the\s+)?(?:\w+\s+){0,3}", 0.7),
            (r"\bwhat\s+(?:are\s+)?(?:the\s+)?(?:impact|effect|implication)", 0.6),
            (r"\bhow\s+does\s+\w+\s+affect\b", 0.7),
            (r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:trend|pattern)", 0.6),
            (r"\bevaluate\s+(?:the\s+)?(?:\w+\s+){0,3}", 0.7),
        ],
        IntentType.COMPARISON: [
            (r"\bvs\.?\s+|\bversus\s+", 0.8),
            (r"\bcompare\s+(?:\w+\s+)?(?:to|with|and)\b", 0.8),
            (r"\bdifference\s+between\b", 0.8),
            (r"\b(?:which|what)\s+is\s+(?:better|worse|best)\b", 0.7),
            (r"\bpro(?:s)?\s+and\s+con(?:s)?\b", 0.8),
            (r"\b\w+\s+or\s+\w+\s*\?", 0.6),
        ],
        IntentType.OPINION: [
            (r"^should\s+i\b", 0.8),
            (r"\bwhat\s+do\s+you\s+(?:think|recommend)\b", 0.8),
            (r"\bis\s+(?:it|this)\s+(?:worth|good|bad)\b", 0.7),
            (r"\brecommend(?:ation)?(?:s)?\s+for\b", 0.7),
            (r"\bsuggestion(?:s)?\s+for\b", 0.7),
        ],
        IntentType.INSTRUCTION: [
            (r"^how\s+(?:to|do\s+(?:i|you)|can\s+(?:i|you))\b", 0.8),
            (r"\bstep(?:s|-)by(?:-step)?\b", 0.8),
            (r"\bguide\s+(?:to|for|on)\b", 0.7),
            (r"\btutorial\s+(?:for|on)\b", 0.8),
            (r"^(?:show|tell)\s+me\s+how\b", 0.7),
        ],
        IntentType.DEFINITION: [
            (r"^what\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?(?:\w+)(?:\s*\?)?$", 0.6),
            (r"\bdefine\s+(?:the\s+)?(?:\w+)\b", 0.8),
            (r"\bmeaning\s+of\s+(?:\w+)\b", 0.8),
            (r"\bexplain\s+(?:what|the)\b", 0.6),
        ],
        IntentType.LISTING: [
            (r"^list\s+(?:all\s+)?(?:the\s+)?(?:\w+)\b", 0.8),
            (r"^(?:what|show)\s+(?:are\s+)?(?:some|all|the)\s+(?:\w+)", 0.7),
            (r"\btop\s+\d+\s+(?:\w+)\b", 0.8),
            (r"\b(?:give|show)\s+(?:me\s+)?(?:\d+\s+)?examples\b", 0.7),
        ],
        IntentType.CAUSAL: [
            (r"^why\s+(?:is|are|did|do|does)\b", 0.8),
            (r"\bwhat\s+(?:caused|causes)\b", 0.8),
            (r"\breason(?:s)?\s+(?:for|why|behind)\b", 0.7),
            (r"\bwhy\s+(?:\w+\s+){0,3}(?:happen|occur)\b", 0.7),
        ],
    }

    def __init__(self):
        """Initialize pattern classifier."""
        # Compile patterns
        self._compiled_patterns: Dict[IntentType, List[Tuple[re.Pattern, float]]] = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                (re.compile(p, re.IGNORECASE), w) for p, w in patterns
            ]

    def classify(self, query: str) -> Dict[IntentType, float]:
        """
        Classify using pattern matching.

        Args:
            query: Input query

        Returns:
            Dict mapping intents to scores
        """
        scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}

        for intent, patterns in self._compiled_patterns.items():
            for pattern, weight in patterns:
                if pattern.search(query):
                    scores[intent] = max(scores[intent], weight)

        return scores


class IntentClassifier:
    """
    Multi-strategy intent classifier.

    Combines rule-based, keyword, and pattern-based classification
    for robust intent detection.

    Example usage:
        >>> classifier = IntentClassifier()
        >>> result = classifier.classify("How do I install Python?")
        >>> print(result.intent)  # IntentType.INSTRUCTION
        >>> print(result.confidence)  # 0.85

        >>> # With multi-label
        >>> result = classifier.classify("Compare pros and cons of React vs Vue")
        >>> print(result.top_intents)  # [COMPARISON, ANALYTICAL]
    """

    def __init__(
        self,
        use_rules: bool = True,
        use_keywords: bool = True,
        use_patterns: bool = True,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize intent classifier.

        Args:
            use_rules: Enable rule-based classification
            use_keywords: Enable keyword-based classification
            use_patterns: Enable pattern-based classification
            confidence_threshold: Minimum confidence for intent detection
        """
        self.confidence_threshold = confidence_threshold

        self._classifiers: List[Tuple[str, Callable[[str], Dict[IntentType, float]], float]] = []

        if use_rules:
            self._classifiers.append(("rules", RuleBasedClassifier().classify, 0.4))
        if use_keywords:
            self._classifiers.append(("keywords", KeywordClassifier().classify, 0.3))
        if use_patterns:
            self._classifiers.append(("patterns", PatternClassifier().classify, 0.3))

        self.extractor = FeatureExtractor()

    def classify(self, query: str) -> IntentResult:
        """
        Classify query intent.

        Args:
            query: Input query string

        Returns:
            IntentResult with classification
        """
        import time

        start_time = time.time()

        # Extract features
        features = self.extractor.extract(query)

        # Determine query type
        query_type = self._determine_query_type(query, features)

        # Aggregate scores from all classifiers
        aggregated_scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}

        for name, classifier, weight in self._classifiers:
            scores = classifier(query)
            for intent, score in scores.items():
                aggregated_scores[intent] += score * weight

        # Normalize
        total_weight = sum(w for _, _, w in self._classifiers)
        if total_weight > 0:
            for intent in aggregated_scores:
                aggregated_scores[intent] /= total_weight

        # Sort by score
        sorted_intents = sorted(
            aggregated_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Get top intent
        top_intent, top_confidence = sorted_intents[0]

        # Filter intents above threshold
        intents_above_threshold = [
            (intent, score)
            for intent, score in sorted_intents
            if score >= self.confidence_threshold
        ]

        if not intents_above_threshold:
            intents_above_threshold = [(IntentType.UNKNOWN, 0.5)]
            top_intent = IntentType.UNKNOWN
            top_confidence = 0.5

        processing_time = (time.time() - start_time) * 1000

        return IntentResult(
            query=query,
            intent=top_intent,
            confidence=top_confidence,
            query_type=query_type,
            intents=intents_above_threshold,
            features=features,
            processing_time_ms=processing_time,
        )

    def _determine_query_type(self, query: str, features: IntentFeatures) -> QueryType:
        """Determine the high-level query type."""
        # Check for question indicators
        if features.has_question_mark:
            return QueryType.QUESTION

        if features.question_word:
            return QueryType.QUESTION

        # Check for command indicators
        command_starters = {
            "show",
            "list",
            "give",
            "find",
            "search",
            "get",
            "tell",
            "explain",
            "describe",
            "define",
            "compare",
        }
        if features.tokens and features.tokens[0] in command_starters:
            return QueryType.COMMAND

        # Check for incomplete/fragment
        if features.word_count < 3:
            return QueryType.INCOMPLETE

        return QueryType.STATEMENT

    def classify_batch(self, queries: List[str]) -> List[IntentResult]:
        """
        Classify multiple queries.

        Args:
            queries: List of queries

        Returns:
            List of IntentResult objects
        """
        return [self.classify(q) for q in queries]

    def get_intent_distribution(self, queries: List[str]) -> Dict[IntentType, int]:
        """
        Get intent distribution for queries.

        Args:
            queries: List of queries

        Returns:
            Dict mapping intents to counts
        """
        distribution: Dict[IntentType, int] = {intent: 0 for intent in IntentType}

        for query in queries:
            result = self.classify(query)
            distribution[result.intent] += 1

        return distribution


# Global classifier instance
_default_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the default intent classifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = IntentClassifier()
    return _default_classifier


def classify_intent(query: str) -> IntentResult:
    """
    Convenience function to classify query intent.

    Args:
        query: Input query

    Returns:
        IntentResult
    """
    return get_intent_classifier().classify(query)


def get_query_intents(query: str) -> List[IntentType]:
    """
    Get all relevant intents for a query.

    Args:
        query: Input query

    Returns:
        List of IntentType
    """
    result = get_intent_classifier().classify(query)
    return result.top_intents


# =============================================================================
# Embedding-Based Classification
# =============================================================================


class EmbeddingClassifier:
    """
    Embedding-based intent classifier.

    Uses cosine similarity with intent exemplars for classification.
    """

    # Default exemplars for each intent
    DEFAULT_EXEMPLARS: Dict[IntentType, List[str]] = {
        IntentType.FACTUAL: [
            "What is machine learning?",
            "Who invented the telephone?",
            "When was Python released?",
            "What is the capital of France?",
        ],
        IntentType.INSTRUCTION: [
            "How do I install Python?",
            "How to create a virtual environment?",
            "What are the steps to deploy a web app?",
            "Guide for setting up Docker",
        ],
        IntentType.COMPARISON: [
            "What's the difference between Python and Java?",
            "Compare React vs Vue",
            "Which is better, SQL or NoSQL?",
            "Pros and cons of monolith vs microservices",
        ],
        IntentType.ANALYTICAL: [
            "Why does Python use indentation?",
            "Explain how neural networks work",
            "What causes memory leaks?",
            "How does garbage collection work?",
        ],
        IntentType.DEFINITION: [
            "What is a neural network?",
            "Define machine learning",
            "Meaning of API",
            "Explain what REST means",
        ],
        IntentType.OPINION: [
            "Should I learn Python or JavaScript?",
            "What do you recommend for web development?",
            "Is React worth learning?",
            "Best programming language for beginners",
        ],
        IntentType.LISTING: [
            "List all programming languages",
            "Show me examples of design patterns",
            "Top 10 Python libraries",
            "What are some machine learning algorithms?",
        ],
        IntentType.CAUSAL: [
            "Why do we need version control?",
            "What causes this error?",
            "Reason for using async programming",
            "Why is Python so popular?",
        ],
    }

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        exemplars: Optional[Dict[IntentType, List[str]]] = None,
    ):
        """Initialize classifier."""
        self.embedder = embedder
        self.exemplars = exemplars or dict(self.DEFAULT_EXEMPLARS)
        self._exemplar_embeddings: Dict[IntentType, List[List[float]]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize exemplar embeddings."""
        if self._initialized or self.embedder is None:
            return

        for intent, texts in self.exemplars.items():
            embeddings = []
            for text in texts:
                try:
                    if asyncio.iscoroutinefunction(self.embedder.embed):
                        embedding = await self.embedder.embed(text)
                    else:
                        embedding = self.embedder.embed(text)  # type: ignore
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed exemplar: {e}")

            self._exemplar_embeddings[intent] = embeddings

        self._initialized = True

    async def classify(
        self,
        query: str,
    ) -> Dict[IntentType, float]:
        """
        Classify query using embeddings.

        Args:
            query: Query text

        Returns:
            Scores for each intent
        """
        await self.initialize()

        if not self._initialized or self.embedder is None:
            return {}

        # Get query embedding
        try:
            if asyncio.iscoroutinefunction(self.embedder.embed):
                query_embedding = await self.embedder.embed(query)
            else:
                query_embedding = self.embedder.embed(query)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return {}

        # Calculate similarity with exemplars
        scores: Dict[IntentType, float] = {}

        for intent, embeddings in self._exemplar_embeddings.items():
            if not embeddings:
                continue

            similarities = [self._cosine_similarity(query_embedding, emb) for emb in embeddings]
            scores[intent] = max(similarities)

        return scores

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Calculate cosine similarity."""
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# =============================================================================
# Async Intent Classifier
# =============================================================================


class AsyncIntentClassifier:
    """
    Async intent classifier with embedding support.

    Combines rule-based, keyword, pattern, and embedding-based classification
    with async operation support.

    Example:
        >>> classifier = AsyncIntentClassifier(embedder)
        >>> result = await classifier.classify("How do I install Python?")
        >>> print(result.intent)  # IntentType.INSTRUCTION
    """

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        use_rules: bool = True,
        use_keywords: bool = True,
        use_patterns: bool = True,
        use_embeddings: bool = True,
        confidence_threshold: float = 0.3,
        embedding_weight: float = 0.4,
    ):
        """
        Initialize async intent classifier.

        Args:
            embedder: Optional embedding model
            use_rules: Enable rule-based classification
            use_keywords: Enable keyword-based classification
            use_patterns: Enable pattern-based classification
            use_embeddings: Enable embedding-based classification
            confidence_threshold: Minimum confidence for intent detection
            embedding_weight: Weight for embedding scores
        """
        self.confidence_threshold = confidence_threshold
        self.embedding_weight = embedding_weight

        # Sync classifiers
        self._sync_classifiers: List[
            Tuple[str, Callable[[str], Dict[IntentType, float]], float]
        ] = []

        total_sync_weight = 0.0
        if use_rules:
            self._sync_classifiers.append(("rules", RuleBasedClassifier().classify, 0.35))
            total_sync_weight += 0.35
        if use_keywords:
            self._sync_classifiers.append(("keywords", KeywordClassifier().classify, 0.25))
            total_sync_weight += 0.25
        if use_patterns:
            self._sync_classifiers.append(("patterns", PatternClassifier().classify, 0.25))
            total_sync_weight += 0.25

        # Normalize sync weights if using embeddings
        if use_embeddings and embedder:
            sync_total = 1.0 - embedding_weight
            if total_sync_weight > 0:
                for i, (name, fn, w) in enumerate(self._sync_classifiers):
                    self._sync_classifiers[i] = (
                        name,
                        fn,
                        w * sync_total / total_sync_weight,
                    )

        # Embedding classifier
        self.embedding_classifier = (
            EmbeddingClassifier(embedder) if use_embeddings and embedder else None
        )

        self.extractor = FeatureExtractor()

    async def classify(self, query: str) -> IntentResult:
        """
        Classify query intent asynchronously.

        Args:
            query: Input query string

        Returns:
            IntentResult with classification
        """
        import time

        start_time = time.time()

        # Extract features
        features = self.extractor.extract(query)

        # Determine query type
        query_type = self._determine_query_type(query, features)

        # Aggregate scores from sync classifiers
        aggregated_scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}

        for _, classifier, weight in self._sync_classifiers:
            scores = classifier(query)
            for intent, score in scores.items():
                aggregated_scores[intent] += score * weight

        # Add embedding scores
        if self.embedding_classifier:
            embedding_scores = await self.embedding_classifier.classify(query)
            for intent, score in embedding_scores.items():
                aggregated_scores[intent] += score * self.embedding_weight

        # Sort by score
        sorted_intents = sorted(
            aggregated_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Get top intent
        top_intent, top_confidence = sorted_intents[0]

        # Filter intents above threshold
        intents_above_threshold = [
            (intent, score)
            for intent, score in sorted_intents
            if score >= self.confidence_threshold
        ]

        if not intents_above_threshold:
            intents_above_threshold = [(IntentType.UNKNOWN, 0.5)]
            top_intent = IntentType.UNKNOWN
            top_confidence = 0.5

        processing_time = (time.time() - start_time) * 1000

        return IntentResult(
            query=query,
            intent=top_intent,
            confidence=min(top_confidence, 1.0),
            query_type=query_type,
            intents=intents_above_threshold,
            features=features,
            processing_time_ms=processing_time,
        )

    def _determine_query_type(self, query: str, features: IntentFeatures) -> QueryType:
        """Determine the high-level query type."""
        if features.has_question_mark:
            return QueryType.QUESTION

        if features.question_word:
            return QueryType.QUESTION

        command_starters = {
            "show",
            "list",
            "give",
            "find",
            "search",
            "get",
            "tell",
            "explain",
            "describe",
            "define",
            "compare",
        }
        if features.tokens and features.tokens[0] in command_starters:
            return QueryType.COMMAND

        if features.word_count < 3:
            return QueryType.INCOMPLETE

        return QueryType.STATEMENT

    async def classify_batch(self, queries: List[str]) -> List[IntentResult]:
        """
        Classify multiple queries asynchronously.

        Args:
            queries: List of queries

        Returns:
            List of IntentResult objects
        """
        return await asyncio.gather(*[self.classify(q) for q in queries])


# =============================================================================
# Intent Router
# =============================================================================


class IntentRouter:
    """
    Routes queries based on intent classification.

    Maps intents to handlers/pipelines for intelligent routing.

    Example:
        >>> router = IntentRouter()
        >>> router.register(IntentType.INSTRUCTION, handle_instruction)
        >>> router.register(IntentType.FACTUAL, handle_factual)
        >>> result, response = await router.route("How do I install Python?")
    """

    def __init__(
        self,
        classifier: Optional[Union[IntentClassifier, AsyncIntentClassifier]] = None,
    ):
        """Initialize router."""
        self.classifier = classifier or IntentClassifier()
        self.routes: Dict[IntentType, Callable] = {}
        self.default_handler: Optional[Callable] = None

    def register(
        self,
        intent: IntentType,
        handler: Callable,
    ) -> None:
        """Register handler for intent."""
        self.routes[intent] = handler

    def set_default(self, handler: Callable) -> None:
        """Set default handler."""
        self.default_handler = handler

    async def route(
        self,
        query: str,
    ) -> Tuple[IntentResult, Any]:
        """
        Route query to appropriate handler.

        Args:
            query: Query text

        Returns:
            Tuple of (classification result, handler result)
        """
        # Classify
        if isinstance(self.classifier, AsyncIntentClassifier):
            result = await self.classifier.classify(query)
        else:
            result = self.classifier.classify(query)

        intent = result.intent

        handler = self.routes.get(intent, self.default_handler)

        if handler is None:
            raise ValueError(f"No handler for intent: {intent}")

        if asyncio.iscoroutinefunction(handler):
            response = await handler(query, result)
        else:
            response = handler(query, result)

        return result, response


# =============================================================================
# Async Convenience Functions
# =============================================================================


async def async_classify_intent(
    query: str,
    embedder: Optional[EmbedderProtocol] = None,
) -> IntentResult:
    """
    Convenience async function to classify query intent.

    Args:
        query: Input query
        embedder: Optional embedding model

    Returns:
        IntentResult
    """
    classifier = AsyncIntentClassifier(embedder=embedder)
    return await classifier.classify(query)
