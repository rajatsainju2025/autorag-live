"""Query Expansion Module for AutoRAG-Live.

Expand queries using various techniques:
- Synonym expansion
- Semantic variations
- Related term generation
- Query decomposition
- Acronym expansion
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExpansionType(Enum):
    """Types of query expansion."""

    SYNONYM = "synonym"
    SEMANTIC = "semantic"
    DECOMPOSITION = "decomposition"
    ACRONYM = "acronym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    RELATED = "related"


@dataclass
class ExpandedTerm:
    """Represents an expanded term."""

    original: str
    expanded: str
    expansion_type: ExpansionType
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpandedQuery:
    """Represents an expanded query."""

    original_query: str
    expanded_queries: list[str] = field(default_factory=list)
    expanded_terms: list[ExpandedTerm] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_queries(self) -> list[str]:
        """Get all query variations."""
        queries = [self.original_query] + self.expanded_queries + self.sub_queries
        return list(dict.fromkeys(queries))  # Remove duplicates while preserving order


@dataclass
class ExpansionConfig:
    """Configuration for query expansion."""

    max_synonyms: int = 3
    max_variations: int = 5
    min_confidence: float = 0.5
    expand_acronyms: bool = True
    decompose_complex: bool = True
    use_stemming: bool = True


class BaseExpander(ABC):
    """Abstract base class for query expanders."""

    expansion_type: ExpansionType

    @abstractmethod
    def expand(
        self,
        query: str,
        config: ExpansionConfig,
    ) -> list[ExpandedTerm]:
        """Expand query terms.

        Args:
            query: Query to expand
            config: Expansion configuration

        Returns:
            List of expanded terms
        """
        pass


class SynonymExpander(BaseExpander):
    """Expand queries using synonyms."""

    expansion_type = ExpansionType.SYNONYM

    # Common synonym groups
    SYNONYM_GROUPS: list[set[str]] = [
        {"find", "search", "look", "locate", "discover"},
        {"show", "display", "present", "demonstrate", "reveal"},
        {"explain", "describe", "clarify", "elaborate", "define"},
        {"create", "make", "build", "generate", "produce"},
        {"delete", "remove", "erase", "eliminate", "discard"},
        {"update", "modify", "change", "alter", "edit"},
        {"get", "retrieve", "fetch", "obtain", "acquire"},
        {"list", "enumerate", "catalog", "itemize", "inventory"},
        {"analyze", "examine", "study", "investigate", "evaluate"},
        {"compare", "contrast", "differentiate", "distinguish"},
        {"important", "significant", "crucial", "vital", "essential"},
        {"problem", "issue", "difficulty", "challenge", "obstacle"},
        {"solution", "answer", "resolution", "fix", "remedy"},
        {"method", "approach", "technique", "procedure", "process"},
        {"example", "instance", "sample", "case", "illustration"},
        {"benefit", "advantage", "gain", "profit", "merit"},
        {"reason", "cause", "factor", "motive", "basis"},
        {"result", "outcome", "effect", "consequence", "impact"},
        {"type", "kind", "sort", "category", "class"},
        {"feature", "characteristic", "attribute", "property", "trait"},
        {"use", "utilize", "employ", "apply", "leverage"},
        {"begin", "start", "commence", "initiate", "launch"},
        {"end", "finish", "complete", "conclude", "terminate"},
        {"include", "contain", "comprise", "encompass", "incorporate"},
        {"improve", "enhance", "boost", "optimize", "upgrade"},
    ]

    def __init__(self) -> None:
        """Initialize synonym expander."""
        # Build lookup dictionary
        self._synonym_lookup: dict[str, set[str]] = {}
        for group in self.SYNONYM_GROUPS:
            for word in group:
                self._synonym_lookup[word] = group - {word}

    def expand(
        self,
        query: str,
        config: ExpansionConfig,
    ) -> list[ExpandedTerm]:
        """Expand query using synonyms."""
        terms: list[ExpandedTerm] = []
        words = query.lower().split()

        for word in words:
            # Clean word
            clean_word = re.sub(r"[^\w]", "", word)
            if not clean_word:
                continue

            synonyms = self._synonym_lookup.get(clean_word, set())

            for i, synonym in enumerate(list(synonyms)[: config.max_synonyms]):
                terms.append(
                    ExpandedTerm(
                        original=clean_word,
                        expanded=synonym,
                        expansion_type=ExpansionType.SYNONYM,
                        confidence=1.0 - (i * 0.1),  # Decrease confidence for later synonyms
                    )
                )

        return terms


class AcronymExpander(BaseExpander):
    """Expand acronyms in queries."""

    expansion_type = ExpansionType.ACRONYM

    # Common technical acronyms
    ACRONYMS: dict[str, str] = {
        # AI/ML
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "llm": "large language model",
        "rag": "retrieval augmented generation",
        "gpt": "generative pre-trained transformer",
        "bert": "bidirectional encoder representations from transformers",
        "gan": "generative adversarial network",
        "cnn": "convolutional neural network",
        "rnn": "recurrent neural network",
        "lstm": "long short-term memory",
        # Development
        "api": "application programming interface",
        "sdk": "software development kit",
        "ide": "integrated development environment",
        "ci": "continuous integration",
        "cd": "continuous deployment",
        "devops": "development operations",
        "orm": "object relational mapping",
        "crud": "create read update delete",
        "rest": "representational state transfer",
        "graphql": "graph query language",
        # Data
        "sql": "structured query language",
        "nosql": "not only structured query language",
        "etl": "extract transform load",
        "olap": "online analytical processing",
        "oltp": "online transaction processing",
        "csv": "comma separated values",
        "json": "javascript object notation",
        "xml": "extensible markup language",
        "yaml": "yet another markup language",
        # Infrastructure
        "aws": "amazon web services",
        "gcp": "google cloud platform",
        "k8s": "kubernetes",
        "vm": "virtual machine",
        "cpu": "central processing unit",
        "gpu": "graphics processing unit",
        "ram": "random access memory",
        "ssd": "solid state drive",
        # Security
        "ssl": "secure sockets layer",
        "tls": "transport layer security",
        "oauth": "open authorization",
        "jwt": "json web token",
        "cors": "cross origin resource sharing",
        # Metrics
        "kpi": "key performance indicator",
        "roi": "return on investment",
        "sla": "service level agreement",
        "mttr": "mean time to recovery",
        "mttf": "mean time to failure",
    }

    def expand(
        self,
        query: str,
        config: ExpansionConfig,
    ) -> list[ExpandedTerm]:
        """Expand acronyms in query."""
        if not config.expand_acronyms:
            return []

        terms: list[ExpandedTerm] = []
        words = query.lower().split()

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word in self.ACRONYMS:
                terms.append(
                    ExpandedTerm(
                        original=clean_word,
                        expanded=self.ACRONYMS[clean_word],
                        expansion_type=ExpansionType.ACRONYM,
                        confidence=0.95,
                    )
                )

        return terms


class QueryDecomposer(BaseExpander):
    """Decompose complex queries into sub-queries."""

    expansion_type = ExpansionType.DECOMPOSITION

    # Patterns for query decomposition
    CONJUNCTION_PATTERNS = [
        r"\band\b",
        r"\bor\b",
        r"\balso\b",
        r"\badditionally\b",
        r"\bas well as\b",
    ]

    MULTI_QUESTION_PATTERNS = [
        r"\?\s*(?:and|what|how|why|when|where|who|which)",
        r"(?:first|second|third|finally|lastly)(?:ly)?[,:]",
    ]

    def expand(
        self,
        query: str,
        config: ExpansionConfig,
    ) -> list[ExpandedTerm]:
        """Decompose complex query into sub-queries."""
        if not config.decompose_complex:
            return []

        sub_queries = self._decompose(query)

        return [
            ExpandedTerm(
                original=query,
                expanded=sub_q,
                expansion_type=ExpansionType.DECOMPOSITION,
                confidence=0.8,
            )
            for sub_q in sub_queries
        ]

    def _decompose(self, query: str) -> list[str]:
        """Decompose query into parts."""
        sub_queries: list[str] = []

        # Split by "and" for compound queries
        and_pattern = re.compile(r"\band\b", re.IGNORECASE)
        parts = and_pattern.split(query)

        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if len(part) > 10:  # Minimum meaningful length
                    sub_queries.append(part)

        # Handle "or" alternatives
        or_pattern = re.compile(r"\bor\b", re.IGNORECASE)
        parts = or_pattern.split(query)

        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if len(part) > 10 and part not in sub_queries:
                    sub_queries.append(part)

        return sub_queries


class RelatedTermExpander(BaseExpander):
    """Expand with related terms and concepts."""

    expansion_type = ExpansionType.RELATED

    # Domain-specific related terms
    RELATED_TERMS: dict[str, list[str]] = {
        # Programming concepts
        "function": ["method", "procedure", "routine", "subroutine"],
        "variable": ["parameter", "argument", "constant", "field"],
        "class": ["object", "type", "struct", "interface"],
        "array": ["list", "collection", "vector", "sequence"],
        "dictionary": ["map", "hashmap", "hashtable", "associative array"],
        "loop": ["iteration", "cycle", "repetition", "traverse"],
        "condition": ["if statement", "branch", "conditional", "predicate"],
        "error": ["exception", "fault", "bug", "failure"],
        "debug": ["troubleshoot", "diagnose", "trace", "inspect"],
        # Data concepts
        "database": ["datastore", "repository", "storage", "db"],
        "query": ["request", "search", "lookup", "fetch"],
        "table": ["relation", "entity", "schema", "dataset"],
        "index": ["key", "pointer", "reference", "lookup"],
        "transaction": ["operation", "commit", "rollback", "atomic"],
        # ML concepts
        "model": ["algorithm", "classifier", "predictor", "estimator"],
        "training": ["learning", "fitting", "optimization", "tuning"],
        "prediction": ["inference", "forecast", "estimation", "output"],
        "accuracy": ["precision", "recall", "f1 score", "performance"],
        "feature": ["attribute", "variable", "dimension", "input"],
        # Document concepts
        "document": ["file", "record", "page", "content"],
        "text": ["content", "passage", "paragraph", "excerpt"],
        "search": ["query", "find", "retrieve", "lookup"],
        "index": ["catalog", "directory", "listing", "inventory"],
    }

    def expand(
        self,
        query: str,
        config: ExpansionConfig,
    ) -> list[ExpandedTerm]:
        """Expand with related terms."""
        terms: list[ExpandedTerm] = []
        words = query.lower().split()

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word in self.RELATED_TERMS:
                for i, related in enumerate(
                    self.RELATED_TERMS[clean_word][: config.max_synonyms]
                ):
                    terms.append(
                        ExpandedTerm(
                            original=clean_word,
                            expanded=related,
                            expansion_type=ExpansionType.RELATED,
                            confidence=0.7 - (i * 0.1),
                        )
                    )

        return terms


class QueryExpansionPipeline:
    """Complete query expansion pipeline."""

    def __init__(
        self,
        config: ExpansionConfig | None = None,
    ) -> None:
        """Initialize expansion pipeline.

        Args:
            config: Expansion configuration
        """
        self.config = config or ExpansionConfig()

        # Initialize expanders
        self._expanders: list[BaseExpander] = [
            SynonymExpander(),
            AcronymExpander(),
            QueryDecomposer(),
            RelatedTermExpander(),
        ]

    def add_expander(self, expander: BaseExpander) -> None:
        """Add custom expander to pipeline.

        Args:
            expander: Expander to add
        """
        self._expanders.append(expander)

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a query using all expanders.

        Args:
            query: Query to expand

        Returns:
            Expanded query result
        """
        expanded_terms: list[ExpandedTerm] = []
        sub_queries: list[str] = []

        # Run all expanders
        for expander in self._expanders:
            try:
                terms = expander.expand(query, self.config)

                for term in terms:
                    if term.confidence >= self.config.min_confidence:
                        if term.expansion_type == ExpansionType.DECOMPOSITION:
                            if term.expanded not in sub_queries:
                                sub_queries.append(term.expanded)
                        else:
                            expanded_terms.append(term)

            except Exception as e:
                logger.warning(f"Expander {expander.expansion_type.value} failed: {e}")

        # Generate expanded query variations
        expanded_queries = self._generate_query_variations(query, expanded_terms)

        return ExpandedQuery(
            original_query=query,
            expanded_queries=expanded_queries[: self.config.max_variations],
            expanded_terms=expanded_terms,
            sub_queries=sub_queries,
        )

    def _generate_query_variations(
        self,
        query: str,
        terms: list[ExpandedTerm],
    ) -> list[str]:
        """Generate query variations using expanded terms."""
        variations: list[str] = []

        query_lower = query.lower()

        # Group terms by original word
        term_groups: dict[str, list[ExpandedTerm]] = {}
        for term in terms:
            if term.original not in term_groups:
                term_groups[term.original] = []
            term_groups[term.original].append(term)

        # Generate single-word substitutions
        for original, group in term_groups.items():
            # Sort by confidence
            sorted_terms = sorted(group, key=lambda t: t.confidence, reverse=True)

            for term in sorted_terms[:2]:  # Top 2 substitutions per word
                # Create variation by replacing the word
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                variation = pattern.sub(term.expanded, query, count=1)

                if variation.lower() != query_lower and variation not in variations:
                    variations.append(variation)

        # Add query with all acronyms expanded
        acronym_expanded = query
        for term in terms:
            if term.expansion_type == ExpansionType.ACRONYM:
                pattern = re.compile(re.escape(term.original), re.IGNORECASE)
                acronym_expanded = pattern.sub(term.expanded, acronym_expanded)

        if acronym_expanded != query and acronym_expanded not in variations:
            variations.append(acronym_expanded)

        return variations

    def expand_for_retrieval(
        self,
        query: str,
        include_original: bool = True,
    ) -> list[str]:
        """Expand query for retrieval (returns list of query strings).

        Args:
            query: Query to expand
            include_original: Include original query

        Returns:
            List of query strings for retrieval
        """
        result = self.expand(query)

        queries = []
        if include_original:
            queries.append(query)

        queries.extend(result.expanded_queries)
        queries.extend(result.sub_queries)

        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries


# Convenience functions


def expand_query(query: str) -> ExpandedQuery:
    """Expand a query using default configuration.

    Args:
        query: Query to expand

    Returns:
        Expanded query result
    """
    pipeline = QueryExpansionPipeline()
    return pipeline.expand(query)


def get_query_variations(
    query: str,
    max_variations: int = 5,
) -> list[str]:
    """Get query variations for retrieval.

    Args:
        query: Query to expand
        max_variations: Maximum variations to return

    Returns:
        List of query variations
    """
    config = ExpansionConfig(max_variations=max_variations)
    pipeline = QueryExpansionPipeline(config)
    return pipeline.expand_for_retrieval(query)


def expand_acronyms(text: str) -> str:
    """Expand acronyms in text.

    Args:
        text: Text with acronyms

    Returns:
        Text with expanded acronyms
    """
    expander = AcronymExpander()
    config = ExpansionConfig()

    terms = expander.expand(text, config)

    result = text
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term.original)}\b", re.IGNORECASE)
        result = pattern.sub(f"{term.expanded} ({term.original})", result)

    return result


def get_synonyms(word: str) -> list[str]:
    """Get synonyms for a word.

    Args:
        word: Word to find synonyms for

    Returns:
        List of synonyms
    """
    expander = SynonymExpander()
    config = ExpansionConfig()

    terms = expander.expand(word, config)
    return [t.expanded for t in terms if t.original == word.lower()]
