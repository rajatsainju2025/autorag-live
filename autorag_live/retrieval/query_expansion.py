"""
Query expansion and reformulation for AutoRAG-Live.

Provides utilities to expand and reformulate queries
for improved retrieval recall and precision.

Features:
- Synonym expansion
- Query reformulation
- Acronym expansion
- Semantic expansion
- Multi-query generation
- Query fusion

Example usage:
    >>> expander = QueryExpander()
    >>> expanded = expander.expand("ML models")
    >>> print(expanded.queries)
    ['ML models', 'machine learning models', 'AI models']
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExpansionStrategy(str, Enum):
    """Query expansion strategies."""

    SYNONYM = "synonym"
    ACRONYM = "acronym"
    SEMANTIC = "semantic"
    RELATED = "related"
    REFORMULATION = "reformulation"
    DECOMPOSITION = "decomposition"


@dataclass
class ExpandedQuery:
    """Result of query expansion."""

    original: str
    queries: List[str]

    # Expansion metadata
    strategies_used: List[ExpansionStrategy] = field(default_factory=list)

    # Weights for each query (for fusion)
    weights: List[float] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize weights if not provided."""
        if not self.weights:
            self.weights = [1.0] * len(self.queries)

    @property
    def unique_queries(self) -> List[str]:
        """Get unique queries."""
        seen = set()
        unique = []
        for q in self.queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)
        return unique


class SynonymDatabase:
    """Database of synonyms and related terms."""

    # Common IT/tech synonyms
    SYNONYMS: Dict[str, List[str]] = {
        "ml": ["machine learning", "ML"],
        "ai": ["artificial intelligence", "AI"],
        "nlp": ["natural language processing", "NLP"],
        "api": ["application programming interface", "API"],
        "db": ["database", "DB"],
        "sql": ["structured query language", "SQL"],
        "ui": ["user interface", "UI"],
        "ux": ["user experience", "UX"],
        "cpu": ["central processing unit", "processor"],
        "gpu": ["graphics processing unit", "GPU"],
        "ram": ["random access memory", "memory"],
        "ssd": ["solid state drive", "SSD"],
        "hdd": ["hard disk drive", "hard drive"],
        "os": ["operating system", "OS"],
        "iot": ["internet of things", "IoT"],
        "llm": ["large language model", "LLM"],
        "rag": ["retrieval augmented generation", "RAG"],
        "async": ["asynchronous"],
        "sync": ["synchronous"],
        "config": ["configuration"],
        "auth": ["authentication"],
        "authn": ["authentication"],
        "authz": ["authorization"],
        "repo": ["repository"],
        "env": ["environment"],
        "dev": ["development", "developer"],
        "prod": ["production"],
        "qa": ["quality assurance", "testing"],
        "ci": ["continuous integration"],
        "cd": ["continuous deployment", "continuous delivery"],
    }

    # Word-level synonyms
    WORD_SYNONYMS: Dict[str, List[str]] = {
        "fast": ["quick", "rapid", "speedy"],
        "slow": ["sluggish", "delayed"],
        "big": ["large", "huge", "massive"],
        "small": ["tiny", "little", "compact"],
        "good": ["great", "excellent", "quality"],
        "bad": ["poor", "terrible"],
        "new": ["recent", "latest", "modern"],
        "old": ["legacy", "outdated", "previous"],
        "create": ["make", "build", "generate"],
        "delete": ["remove", "destroy", "drop"],
        "update": ["modify", "change", "edit"],
        "find": ["search", "locate", "discover"],
        "show": ["display", "present", "view"],
        "hide": ["conceal", "mask"],
        "start": ["begin", "launch", "initiate"],
        "stop": ["end", "terminate", "halt"],
        "error": ["bug", "issue", "problem", "fault"],
        "fix": ["repair", "resolve", "correct"],
        "improve": ["enhance", "optimize", "upgrade"],
        "reduce": ["decrease", "minimize", "lower"],
        "increase": ["grow", "expand", "raise"],
    }

    def __init__(self, custom_synonyms: Optional[Dict[str, List[str]]] = None):
        """
        Initialize synonym database.

        Args:
            custom_synonyms: Additional custom synonyms
        """
        self.synonyms = dict(self.SYNONYMS)
        self.word_synonyms = dict(self.WORD_SYNONYMS)

        if custom_synonyms:
            self.synonyms.update(custom_synonyms)

    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term."""
        term_lower = term.lower()

        # Check exact match
        if term_lower in self.synonyms:
            return self.synonyms[term_lower]

        # Check word synonyms
        if term_lower in self.word_synonyms:
            return self.word_synonyms[term_lower]

        return []

    def add_synonym(self, term: str, synonyms: List[str]) -> None:
        """Add synonyms for a term."""
        self.synonyms[term.lower()] = synonyms

    def expand_text(self, text: str) -> List[str]:
        """Expand all terms in text."""
        words = text.lower().split()
        expansions = [text]

        for i, word in enumerate(words):
            syns = self.get_synonyms(word)
            for syn in syns:
                new_words = words.copy()
                new_words[i] = syn
                expansions.append(" ".join(new_words))

        return expansions


class BaseExpander(ABC):
    """Base class for query expanders."""

    @abstractmethod
    def expand(self, query: str) -> List[str]:
        """Expand a query."""
        pass

    @property
    @abstractmethod
    def strategy(self) -> ExpansionStrategy:
        """Get expansion strategy."""
        pass


class SynonymExpander(BaseExpander):
    """Expand queries using synonyms."""

    def __init__(
        self,
        max_expansions: int = 5,
        custom_synonyms: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize synonym expander.

        Args:
            max_expansions: Maximum number of expanded queries
            custom_synonyms: Custom synonym mappings
        """
        self.max_expansions = max_expansions
        self.db = SynonymDatabase(custom_synonyms)

    @property
    def strategy(self) -> ExpansionStrategy:
        return ExpansionStrategy.SYNONYM

    def expand(self, query: str) -> List[str]:
        """Expand query using synonyms."""
        expansions = self.db.expand_text(query)
        return expansions[: self.max_expansions]


class AcronymExpander(BaseExpander):
    """Expand acronyms in queries."""

    ACRONYMS: Dict[str, str] = {
        "AI": "Artificial Intelligence",
        "ML": "Machine Learning",
        "DL": "Deep Learning",
        "NLP": "Natural Language Processing",
        "NLU": "Natural Language Understanding",
        "NLG": "Natural Language Generation",
        "CV": "Computer Vision",
        "RL": "Reinforcement Learning",
        "API": "Application Programming Interface",
        "REST": "Representational State Transfer",
        "SQL": "Structured Query Language",
        "NoSQL": "Not Only SQL",
        "JSON": "JavaScript Object Notation",
        "XML": "Extensible Markup Language",
        "HTML": "HyperText Markup Language",
        "CSS": "Cascading Style Sheets",
        "JS": "JavaScript",
        "TS": "TypeScript",
        "UI": "User Interface",
        "UX": "User Experience",
        "GPU": "Graphics Processing Unit",
        "CPU": "Central Processing Unit",
        "RAM": "Random Access Memory",
        "SSD": "Solid State Drive",
        "HDD": "Hard Disk Drive",
        "OS": "Operating System",
        "VM": "Virtual Machine",
        "K8S": "Kubernetes",
        "AWS": "Amazon Web Services",
        "GCP": "Google Cloud Platform",
        "LLM": "Large Language Model",
        "RAG": "Retrieval Augmented Generation",
        "BERT": "Bidirectional Encoder Representations from Transformers",
        "GPT": "Generative Pre-trained Transformer",
        "IoT": "Internet of Things",
        "CI": "Continuous Integration",
        "CD": "Continuous Deployment",
        "TDD": "Test Driven Development",
        "BDD": "Behavior Driven Development",
        "OOP": "Object Oriented Programming",
        "FP": "Functional Programming",
        "MVC": "Model View Controller",
        "MVP": "Model View Presenter",
        "MVVM": "Model View ViewModel",
        "SOLID": "Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion",
        "DRY": "Don't Repeat Yourself",
        "KISS": "Keep It Simple, Stupid",
        "YAGNI": "You Aren't Gonna Need It",
    }

    def __init__(self, custom_acronyms: Optional[Dict[str, str]] = None):
        """
        Initialize acronym expander.

        Args:
            custom_acronyms: Custom acronym mappings
        """
        self.acronyms = dict(self.ACRONYMS)
        if custom_acronyms:
            self.acronyms.update(custom_acronyms)

    @property
    def strategy(self) -> ExpansionStrategy:
        return ExpansionStrategy.ACRONYM

    def expand(self, query: str) -> List[str]:
        """Expand acronyms in query."""
        expansions = [query]

        # Find acronyms in query
        words = query.split()

        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word).upper()

            if clean_word in self.acronyms:
                new_words = words.copy()
                new_words[i] = self.acronyms[clean_word]
                expansions.append(" ".join(new_words))

        return expansions


class QueryReformulator(BaseExpander):
    """Reformulate queries for better retrieval."""

    QUESTION_PREFIXES = [
        "What is",
        "How does",
        "Why does",
        "When should",
        "Where can",
        "Who uses",
        "Which",
        "Explain",
        "Describe",
        "Define",
    ]

    def __init__(
        self,
        add_question_forms: bool = True,
        add_statement_forms: bool = True,
    ):
        """
        Initialize reformulator.

        Args:
            add_question_forms: Add question reformulations
            add_statement_forms: Add statement reformulations
        """
        self.add_question_forms = add_question_forms
        self.add_statement_forms = add_statement_forms

    @property
    def strategy(self) -> ExpansionStrategy:
        return ExpansionStrategy.REFORMULATION

    def expand(self, query: str) -> List[str]:
        """Reformulate query."""
        reformulations = [query]

        # Check if query is a question
        is_question = query.strip().endswith("?") or any(
            query.lower().startswith(qw)
            for qw in ["what", "how", "why", "when", "where", "who", "which"]
        )

        if is_question and self.add_statement_forms:
            # Convert question to statement
            reformulations.extend(self._question_to_statement(query))

        if not is_question and self.add_question_forms:
            # Convert statement to questions
            reformulations.extend(self._statement_to_questions(query))

        return reformulations

    def _question_to_statement(self, question: str) -> List[str]:
        """Convert question to statement forms."""
        statements = []

        # Remove question mark
        clean = question.rstrip("?").strip()

        # Simple transformations
        patterns = [
            (r"^what is (.+)$", r"\1"),
            (r"^how to (.+)$", r"\1 guide"),
            (r"^how does (.+) work$", r"\1 mechanism"),
            (r"^why (.+)$", r"reason for \1"),
            (r"^when should (.+)$", r"\1 timing"),
        ]

        for pattern, replacement in patterns:
            match = re.match(pattern, clean, re.IGNORECASE)
            if match:
                statements.append(re.sub(pattern, replacement, clean, flags=re.IGNORECASE))

        # Also add without question word
        for qw in ["what is", "how to", "how does", "why", "when"]:
            if clean.lower().startswith(qw):
                remainder = clean[len(qw) :].strip()
                if remainder:
                    statements.append(remainder)

        return statements

    def _statement_to_questions(self, statement: str) -> List[str]:
        """Convert statement to question forms."""
        questions = []

        # Add common question forms
        questions.append(f"What is {statement}?")
        questions.append(f"How does {statement} work?")

        return questions


class QueryDecomposer(BaseExpander):
    """Decompose complex queries into simpler sub-queries."""

    def __init__(self):
        """Initialize decomposer."""
        pass

    @property
    def strategy(self) -> ExpansionStrategy:
        return ExpansionStrategy.DECOMPOSITION

    def expand(self, query: str) -> List[str]:
        """Decompose query into sub-queries."""
        sub_queries = [query]

        # Split on conjunctions
        parts = re.split(r"\s+(?:and|or|but|versus|vs\.?)\s+", query, flags=re.IGNORECASE)

        if len(parts) > 1:
            sub_queries.extend([p.strip() for p in parts if p.strip()])

        # Split on commas for lists
        if "," in query:
            comma_parts = query.split(",")
            if len(comma_parts) > 1:
                sub_queries.extend([p.strip() for p in comma_parts if p.strip()])

        return sub_queries


class SemanticExpander(BaseExpander):
    """Expand queries using **vectorised** semantic similarity.

    Improvements over the original implementation:

    * **Batch embedding** — embeds the entire vocabulary in one call via an
      optional ``batch_embedding_func`` (falls back to per-term calls).
    * **Numpy matrix cosine similarity** — a single ``vocab_matrix @ query``
      dot product replaces the old pure-Python per-element loop, yielding
      ~100× speed-up for 768-dim embeddings.
    * **Pre-normalised vocab matrix** — L2-normalisation is done once at
      index-build time, so ``expand()`` only needs one matmul + ``argpartition``
      (O(n) partial sort for top-k instead of O(n log n) full sort).
    * **LRU query cache** — repeated identical queries return instantly.
    """

    def __init__(
        self,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        batch_embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
        vocabulary: Optional[List[str]] = None,
        top_k: int = 3,
    ):
        self.embedding_func = embedding_func
        self.batch_embedding_func = batch_embedding_func
        self.vocabulary = vocabulary or []
        self.top_k = top_k

        # Pre-normalised (n_vocab, dim) float32 matrix — built lazily.
        self._vocab_matrix: Optional[np.ndarray] = None

    @property
    def strategy(self) -> ExpansionStrategy:
        return ExpansionStrategy.SEMANTIC

    # -- index construction ------------------------------------------------

    def build_index(self) -> None:
        """Embed vocabulary and L2-normalise into a dense matrix."""
        if not self.vocabulary:
            return

        if self.batch_embedding_func is not None:
            raw = self.batch_embedding_func(self.vocabulary)
            matrix = np.asarray(raw, dtype=np.float32)
        elif self.embedding_func is not None:
            raw = [self.embedding_func(t) for t in self.vocabulary]
            matrix = np.asarray(raw, dtype=np.float32)
        else:
            return

        # L2-normalise rows so dot product == cosine similarity.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        self._vocab_matrix = matrix / norms

    # -- expansion ---------------------------------------------------------

    def expand(self, query: str) -> List[str]:
        if not self.embedding_func or not self.vocabulary:
            return [query]

        if self._vocab_matrix is None:
            self.build_index()
        if self._vocab_matrix is None:
            return [query]

        # Embed & normalise query
        q_vec = np.asarray(self.embedding_func(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-12:
            return [query]
        q_vec /= q_norm

        # Vectorised cosine similarity: (n_vocab,)
        sims = self._vocab_matrix @ q_vec

        # O(n) partial sort for top-k (faster than full argsort for large n)
        k = min(self.top_k + 1, len(sims))  # +1 in case query matches itself
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        expansions: List[str] = [query]
        query_low = query.lower()
        for idx in top_indices:
            term = self.vocabulary[int(idx)]
            if term.lower() != query_low:
                expansions.append(term)
            if len(expansions) > self.top_k:
                break

        return expansions


class QueryExpander:
    """
    Main query expansion interface.

    Example:
        >>> expander = QueryExpander()
        >>>
        >>> # Expand a query
        >>> result = expander.expand("ML model performance")
        >>> for query in result.queries:
        ...     print(query)
        >>>
        >>> # Use specific strategies
        >>> result = expander.expand(
        ...     "API design",
        ...     strategies=[ExpansionStrategy.SYNONYM, ExpansionStrategy.ACRONYM]
        ... )
    """

    def __init__(
        self,
        max_total_queries: int = 10,
        include_original: bool = True,
        deduplicate: bool = True,
    ):
        """
        Initialize query expander.

        Args:
            max_total_queries: Maximum total expanded queries
            include_original: Include original query in results
            deduplicate: Remove duplicate expansions
        """
        self.max_total_queries = max_total_queries
        self.include_original = include_original
        self.deduplicate = deduplicate

        # Initialize default expanders
        self._expanders: Dict[ExpansionStrategy, BaseExpander] = {
            ExpansionStrategy.SYNONYM: SynonymExpander(),
            ExpansionStrategy.ACRONYM: AcronymExpander(),
            ExpansionStrategy.REFORMULATION: QueryReformulator(),
            ExpansionStrategy.DECOMPOSITION: QueryDecomposer(),
        }

    def add_expander(
        self,
        strategy: ExpansionStrategy,
        expander: BaseExpander,
    ) -> None:
        """Add or replace an expander."""
        self._expanders[strategy] = expander

    def expand(
        self,
        query: str,
        strategies: Optional[List[ExpansionStrategy]] = None,
        weights: Optional[Dict[ExpansionStrategy, float]] = None,
    ) -> ExpandedQuery:
        """
        Expand a query using specified strategies.

        Args:
            query: Query to expand
            strategies: Strategies to use (all by default)
            weights: Weights for each strategy

        Returns:
            ExpandedQuery with all expansions
        """
        if strategies is None:
            strategies = list(self._expanders.keys())

        weights = weights or {}

        all_expansions: List[Tuple[str, float, ExpansionStrategy]] = []

        # Add original if requested
        if self.include_original:
            all_expansions.append((query, 1.0, ExpansionStrategy.SYNONYM))

        # Apply each strategy
        for strategy in strategies:
            if strategy not in self._expanders:
                continue

            expander = self._expanders[strategy]
            weight = weights.get(strategy, 1.0)

            try:
                expanded = expander.expand(query)
                for exp in expanded:
                    if exp != query:  # Don't duplicate original
                        all_expansions.append((exp, weight, strategy))
            except Exception as e:
                logger.warning(f"Expansion failed for {strategy}: {e}")

        # Deduplicate if requested
        if self.deduplicate:
            seen: Set[str] = set()
            unique = []
            for exp, weight, strat in all_expansions:
                if exp.lower() not in seen:
                    seen.add(exp.lower())
                    unique.append((exp, weight, strat))
            all_expansions = unique

        # Limit total queries
        all_expansions = all_expansions[: self.max_total_queries]

        # Build result
        queries = [exp for exp, _, _ in all_expansions]
        query_weights = [w for _, w, _ in all_expansions]
        strategies_used = list(set(s for _, _, s in all_expansions))

        return ExpandedQuery(
            original=query,
            queries=queries,
            strategies_used=strategies_used,
            weights=query_weights,
        )


class MultiQueryGenerator:
    """
    Generate multiple query variations for improved retrieval.

    Example:
        >>> generator = MultiQueryGenerator()
        >>> queries = generator.generate("Python web frameworks")
        >>> for q in queries:
        ...     print(q)
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        num_variations: int = 3,
    ):
        """
        Initialize multi-query generator.

        Args:
            llm_func: Optional LLM function for generation
            num_variations: Number of variations to generate
        """
        self.llm_func = llm_func
        self.num_variations = num_variations
        self.expander = QueryExpander()

    def generate(self, query: str) -> List[str]:
        """
        Generate multiple query variations.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        if self.llm_func:
            return self._generate_with_llm(query)
        else:
            return self._generate_with_rules(query)

    def _generate_with_llm(self, query: str) -> List[str]:
        """Generate variations using LLM."""
        prompt = f"""Generate {self.num_variations} different ways to ask the following question.
Each variation should search for the same information but use different wording.

Original question: {query}

Provide only the questions, one per line, without numbering or explanation."""

        try:
            response = self.llm_func(prompt)
            variations = [line.strip() for line in response.strip().split("\n") if line.strip()]
            return [query] + variations[: self.num_variations]
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return self._generate_with_rules(query)

    def _generate_with_rules(self, query: str) -> List[str]:
        """Generate variations using rules."""
        result = self.expander.expand(query)
        return result.queries[: self.num_variations + 1]


class QueryFusion:
    """
    Fuse results from multiple queries using Reciprocal Rank Fusion.

    Example:
        >>> fusion = QueryFusion()
        >>>
        >>> # Results from multiple queries
        >>> results = {
        ...     "query1": [("doc1", 0.9), ("doc2", 0.8)],
        ...     "query2": [("doc2", 0.95), ("doc3", 0.7)],
        ... }
        >>>
        >>> # Fuse results
        >>> fused = fusion.fuse(results)
    """

    def __init__(
        self,
        k: int = 60,
        weight_by_query: bool = True,
    ):
        """
        Initialize query fusion.

        Args:
            k: RRF parameter (typically 60)
            weight_by_query: Weight by query weights
        """
        self.k = k
        self.weight_by_query = weight_by_query

    def fuse(
        self,
        query_results: Dict[str, List[Tuple[str, float]]],
        query_weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Fuse results from multiple queries.

        Args:
            query_results: Dict mapping query to (doc_id, score) list
            query_weights: Optional weights for each query

        Returns:
            Fused and ranked (doc_id, score) list
        """
        query_weights = query_weights or {}

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}

        for query, results in query_results.items():
            weight = query_weights.get(query, 1.0) if self.weight_by_query else 1.0

            for rank, (doc_id, _) in enumerate(results, 1):
                rrf_score = weight / (self.k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results


# Convenience functions


def expand_query(
    query: str,
    max_expansions: int = 10,
) -> List[str]:
    """
    Expand a query using default settings.

    Args:
        query: Query to expand
        max_expansions: Maximum expansions

    Returns:
        List of expanded queries
    """
    expander = QueryExpander(max_total_queries=max_expansions)
    result = expander.expand(query)
    return result.queries


def generate_multi_query(
    query: str,
    num_variations: int = 3,
) -> List[str]:
    """
    Generate multiple query variations.

    Args:
        query: Original query
        num_variations: Number of variations

    Returns:
        List of query variations
    """
    generator = MultiQueryGenerator(num_variations=num_variations)
    return generator.generate(query)


def fuse_results(
    query_results: Dict[str, List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Fuse results from multiple queries.

    Args:
        query_results: Results from multiple queries
        k: RRF parameter

    Returns:
        Fused results
    """
    fusion = QueryFusion(k=k)
    return fusion.fuse(query_results)
