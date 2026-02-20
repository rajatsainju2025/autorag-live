"""
Query-Adaptive Generation Parameter Tuner.

Dynamically selects LLM sampling hyperparameters (temperature, top-p,
top-k, max_tokens, presence/frequency penalty) based on the query's
semantic type and complexity — instead of using static defaults.

Rationale:
    - **Factoid / exact-answer queries** (e.g. "What year was Python created?")
      → temperature=0.0, top-p=0.9: greedy decoding for precision.
    - **Creative / open-ended queries** (e.g. "Generate ideas for …")
      → temperature=0.9, top-p=0.95: diverse sampling for variety.
    - **Analytical / reasoning queries** (e.g. "Compare X and Y")
      → temperature=0.3, top-p=0.92: controlled creativity for coherent reasoning.
    - **High-complexity multi-hop queries**
      → increased max_tokens budget to allow full chain-of-thought.

The module integrates with the existing ``QueryAnalysis`` dataclass from
``autorag_live.routing.router`` and can be used standalone.

References:
- "On the Calibration of LLMs" (Zhao et al., 2023)
- "Do Not Trust ChatGPT When Asked" (Pan et al., 2023) — static temp harms
  factoid accuracy; adaptive sampling helps.
- OpenAI API docs: temperature, top_p guidance (2024).

Example:
    >>> tuner = AdaptiveParamTuner()
    >>> params = tuner.tune("What is the speed of light?")
    >>> print(params)
    GenerationParams(temperature=0.0, top_p=0.9, max_tokens=256, ...)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query-type classification
# ---------------------------------------------------------------------------


class QueryIntent(str, Enum):
    """Semantic intent categories for generation parameter selection."""

    FACTOID = "factoid"
    """Single-answer lookup: 'What is X?', 'When was Y?'"""

    PROCEDURAL = "procedural"
    """Step-by-step instructions: 'How do I…?', 'Steps to…'"""

    ANALYTICAL = "analytical"
    """Reasoning / comparison: 'Compare X and Y', 'Why does…', 'Analyse…'"""

    CREATIVE = "creative"
    """Open-ended generation: 'Generate…', 'Write…', 'Create…'"""

    CONVERSATIONAL = "conversational"
    """Dialogue continuation: greetings, follow-ups, clarifications."""

    UNKNOWN = "unknown"
    """Default when intent is not classifiable."""


# ── Regex classifiers (order matters — first match wins) ──────────────────

_INTENT_PATTERNS: list[tuple[QueryIntent, re.Pattern]] = [
    (
        QueryIntent.FACTOID,
        re.compile(
            r"^(what|who|where|when|which|how many|how much|is|are|was|were|"
            r"does|did|name|list|define|definition|meaning of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QueryIntent.PROCEDURAL,
        re.compile(
            r"^(how (do|to|can|should|would)|steps|guide|tutorial|walk me through|"
            r"instructions|explain how|process of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QueryIntent.ANALYTICAL,
        re.compile(
            r"^(compare|contrast|analyse|analyze|evaluate|assess|why|reason|"
            r"explain (why|the reason)|what are (the pros|the cons|the advantages)|"
            r"differences? between|similarities? between)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QueryIntent.CREATIVE,
        re.compile(
            r"^(generate|create|write|draft|compose|brainstorm|suggest|come up with|"
            r"imagine|design|ideas? for|story|poem|script)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QueryIntent.CONVERSATIONAL,
        re.compile(
            r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|tell me|chat)\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_intent(query: str) -> QueryIntent:
    """
    Classify the semantic intent of a query.

    Args:
        query: Raw query string.

    Returns:
        QueryIntent enum value.
    """
    q = query.strip()
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(q):
            return intent
    return QueryIntent.UNKNOWN


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------


def estimate_complexity(query: str) -> float:
    """
    Estimate query complexity as a score in [0, 1].

    Factors:
    - Word count (longer = more complex)
    - Presence of multi-hop indicators ('and then', 'given that', 'assuming')
    - Number of question marks (compound questions)
    - Named entity density (digits, capitalised words)

    Returns:
        Float in [0.0, 1.0].
    """
    words = query.split()
    n_words = len(words)

    # Word-count contribution (normalised, cap at 40 words)
    word_score = min(n_words / 40.0, 1.0)

    # Multi-hop keywords
    multi_hop_kws = re.findall(
        r"\b(and then|after that|given that|assuming|furthermore|in addition|"
        r"therefore|consequently|as a result|first.*then|step|multi|multiple)\b",
        query,
        re.IGNORECASE,
    )
    hop_score = min(len(multi_hop_kws) / 3.0, 1.0)

    # Compound questions
    q_count = query.count("?")
    q_score = min(q_count / 3.0, 1.0)

    # Named-entity density
    cap_words = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
    entity_score = min(cap_words / max(n_words, 1), 1.0)

    return 0.4 * word_score + 0.3 * hop_score + 0.2 * q_score + 0.1 * entity_score


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------


@dataclass
class GenerationParams:
    """
    Complete set of LLM generation hyperparameters.

    Compatible with OpenAI, Anthropic Claude, Cohere, and Mistral APIs.
    """

    temperature: float = 0.3
    """Sampling temperature ∈ [0, 2]. 0 = greedy, >1 = more random."""

    top_p: float = 0.92
    """Nucleus sampling probability mass ∈ (0, 1]."""

    top_k: int = 50
    """Top-k sampling (set to 0 to disable for APIs that support it)."""

    max_tokens: int = 512
    """Maximum tokens to generate."""

    presence_penalty: float = 0.0
    """Penalises new tokens if they appeared in the context (OpenAI-style)."""

    frequency_penalty: float = 0.0
    """Penalises tokens proportional to their frequency so far."""

    stop_sequences: list[str] = field(default_factory=list)
    """Optional stop sequences."""

    intent: QueryIntent = QueryIntent.UNKNOWN
    """The classified intent that produced these parameters."""

    complexity: float = 0.5
    """Estimated query complexity that influenced max_tokens."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to API-compatible dict (drops intent/complexity metadata)."""
        d = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.presence_penalty:
            d["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty:
            d["frequency_penalty"] = self.frequency_penalty
        if self.stop_sequences:
            d["stop"] = self.stop_sequences
        return d

    def to_anthropic_dict(self) -> Dict[str, Any]:
        """Serialise for Anthropic Claude API format."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
            "max_tokens": self.max_tokens,
        }


# ---------------------------------------------------------------------------
# Adaptive Tuner
# ---------------------------------------------------------------------------


# Base parameter presets per intent
_INTENT_PRESETS: Dict[QueryIntent, Dict[str, Any]] = {
    QueryIntent.FACTOID: {
        "temperature": 0.0,
        "top_p": 0.90,
        "top_k": 1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "base_max_tokens": 256,
    },
    QueryIntent.PROCEDURAL: {
        "temperature": 0.2,
        "top_p": 0.92,
        "top_k": 40,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "base_max_tokens": 512,
    },
    QueryIntent.ANALYTICAL: {
        "temperature": 0.3,
        "top_p": 0.93,
        "top_k": 50,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.0,
        "base_max_tokens": 768,
    },
    QueryIntent.CREATIVE: {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 100,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "base_max_tokens": 1024,
    },
    QueryIntent.CONVERSATIONAL: {
        "temperature": 0.7,
        "top_p": 0.93,
        "top_k": 50,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.0,
        "base_max_tokens": 256,
    },
    QueryIntent.UNKNOWN: {
        "temperature": 0.3,
        "top_p": 0.92,
        "top_k": 50,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "base_max_tokens": 512,
    },
}


class AdaptiveParamTuner:
    """
    Selects LLM generation hyperparameters from query intent + complexity.

    Usage:
        >>> tuner = AdaptiveParamTuner(max_tokens_ceiling=2048)
        >>> params = tuner.tune("Why did the Roman Empire fall?")
        >>> print(params.temperature, params.max_tokens)

    Args:
        max_tokens_ceiling: Absolute max tokens cap (default 2048).
        complexity_token_scale: Extra tokens per complexity unit (default 512).
        override_presets: Dict to override any preset values per intent.
    """

    def __init__(
        self,
        max_tokens_ceiling: int = 2048,
        complexity_token_scale: int = 512,
        override_presets: Optional[Dict[QueryIntent, Dict[str, Any]]] = None,
    ) -> None:
        self.max_tokens_ceiling = max_tokens_ceiling
        self.complexity_token_scale = complexity_token_scale
        # Deep copy presets and apply any overrides
        import copy

        self._presets = copy.deepcopy(_INTENT_PRESETS)
        if override_presets:
            for intent, overrides in override_presets.items():
                self._presets[intent].update(overrides)

    def tune(
        self,
        query: str,
        intent: Optional[QueryIntent] = None,
        complexity: Optional[float] = None,
    ) -> GenerationParams:
        """
        Compute optimal generation parameters for a query.

        Args:
            query: The user query string.
            intent: Pre-computed intent (auto-classifies if None).
            complexity: Pre-computed complexity ∈ [0,1] (auto-estimates if None).

        Returns:
            GenerationParams instance.
        """
        resolved_intent = intent or classify_intent(query)
        resolved_complexity = complexity if complexity is not None else estimate_complexity(query)

        preset = self._presets[resolved_intent]

        # Scale max_tokens by complexity: more complex → more tokens
        base_max = preset["base_max_tokens"]
        scaled_max = int(base_max + resolved_complexity * self.complexity_token_scale)
        final_max_tokens = min(scaled_max, self.max_tokens_ceiling)

        params = GenerationParams(
            temperature=preset["temperature"],
            top_p=preset["top_p"],
            top_k=preset["top_k"],
            max_tokens=final_max_tokens,
            presence_penalty=preset["presence_penalty"],
            frequency_penalty=preset["frequency_penalty"],
            intent=resolved_intent,
            complexity=round(resolved_complexity, 3),
        )

        logger.debug(
            "AdaptiveParams: intent=%s complexity=%.2f → temp=%.2f top_p=%.2f max_tokens=%d",
            resolved_intent.value,
            resolved_complexity,
            params.temperature,
            params.top_p,
            params.max_tokens,
        )
        return params

    def tune_for_consistency(
        self,
        query: str,
        n_samples: int = 5,
    ) -> GenerationParams:
        """
        Parameters optimised for self-consistency sampling.

        Locks temperature to 0.7 and adjusts top_p slightly upward
        to ensure sufficient sample diversity for majority voting.

        Args:
            query: Query string.
            n_samples: Number of consistency samples planned (unused directly
                       but surfaced in metadata for downstream observability).

        Returns:
            GenerationParams with consistency-friendly settings.
        """
        params = self.tune(query)
        # Override for consistency sampling
        params.temperature = 0.7
        params.top_p = min(params.top_p + 0.03, 1.0)
        params.metadata = {"consistency_samples": n_samples}  # type: ignore[attr-defined]
        return params
