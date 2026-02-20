"""
GraphRAG Community Summarizer.

Implements the community detection + hierarchical LLM summarization
component of Microsoft's GraphRAG framework (Edge et al., 2024).

Pipeline
--------
1. **Entity co-occurrence graph** — build a weighted graph where edge
   weight = number of documents in which two entities co-occur.
2. **Louvain community detection** — partition entities into communities
   using a pure-numpy modularity-optimising algorithm (no networkx dep).
3. **Hierarchical summarization** — for each community, call the LLM to
   produce a concise community summary from its member entities and
   their shared evidence.
4. **Global query** — answer corpus-level "What are the main themes?"
   questions by retrieving and assembling relevant community summaries.
5. **Local query** — answer entity-specific questions by traversing the
   graph and collecting entity + community context.

References
----------
- "From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization" Edge et al., 2024 (https://arxiv.org/abs/2404.16130)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GraphEntity:
    """An entity in the co-occurrence graph."""

    name: str
    entity_type: str = "unknown"
    description: str = ""
    mentions: List[str] = field(default_factory=list)  # source texts

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphEntity):
            return NotImplemented
        return self.name.lower() == other.name.lower()


@dataclass
class Community:
    """A cluster of related entities."""

    community_id: str
    entities: List[GraphEntity] = field(default_factory=list)
    summary: str = ""
    level: int = 0  # 0 = leaf, higher = more abstract
    child_community_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    @property
    def entity_names(self) -> List[str]:
        return [e.name for e in self.entities]


@dataclass
class CommunitySummarizerResult:
    """Result of community summarisation for a query."""

    query: str
    answer: str
    relevant_communities: List[Community]
    query_type: str  # "global" or "local"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure-numpy Louvain (simplified greedy modularity optimisation)
# ---------------------------------------------------------------------------


def _louvain_communities(
    adjacency: np.ndarray,
    resolution: float = 1.0,
    max_iter: int = 100,
    seed: int = 42,
) -> List[List[int]]:
    """
    Simplified greedy Louvain community detection.

    Operates on a symmetric adjacency matrix.  Returns a list of
    communities, each a list of node indices.

    This is a single-phase Louvain (no phase-2 super-node contraction),
    which is sufficient for the entity graph sizes found in typical RAG
    corpora (< 10 000 entities).

    Args:
        adjacency: (n, n) symmetric weight matrix.
        resolution: Modularity resolution parameter (higher → smaller communities).
        max_iter: Max greedy optimisation passes.
        seed: Random seed for reproducibility.

    Returns:
        List of community lists (node indices per community).
    """
    rng = random.Random(seed)
    n = adjacency.shape[0]
    if n == 0:
        return []

    # Initialise: each node in its own community
    labels = list(range(n))
    m = adjacency.sum() / 2.0  # total edge weight

    if m == 0:
        return [[i] for i in range(n)]

    # Degree vector
    degree = adjacency.sum(axis=1)

    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        order = list(range(n))
        rng.shuffle(order)

        for i in order:
            current_label = labels[i]

            # Neighbour communities
            neighbour_labels: Dict[int, float] = defaultdict(float)
            for j in range(n):
                if adjacency[i, j] > 0 and i != j:
                    neighbour_labels[labels[j]] += adjacency[i, j]

            if not neighbour_labels:
                continue

            # Modularity gain for moving i to each neighbour community
            best_label = current_label
            best_gain = 0.0

            for label, w_in in neighbour_labels.items():
                if label == current_label:
                    continue
                # Sum of degrees in candidate community
                sum_deg = sum(degree[j] for j in range(n) if labels[j] == label)
                gain = w_in - resolution * (sum_deg * degree[i]) / (2.0 * m)
                if gain > best_gain:
                    best_gain = gain
                    best_label = label

            if best_label != current_label:
                labels[i] = best_label
                improved = True

        iterations += 1

    # Group by label
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Summarisation prompts
# ---------------------------------------------------------------------------

_COMMUNITY_SUMMARY_PROMPT = (
    "You are summarising a community of related entities from a knowledge graph.\n\n"
    "Community entities: {entities}\n\n"
    "Evidence / context:\n{evidence}\n\n"
    "Write a concise 2–3 sentence summary of what this community represents, "
    "what connects these entities, and why they are important in the corpus. "
    "Focus on factual relationships."
)

_GLOBAL_ANSWER_PROMPT = (
    "You are answering a global, summary-level question about a document corpus.\n\n"
    "Query: {query}\n\n"
    "Community summaries:\n{summaries}\n\n"
    "Using the community summaries above, write a comprehensive answer to the query."
)

_LOCAL_ANSWER_PROMPT = (
    "You are answering a specific question using entity graph context.\n\n"
    "Query: {query}\n\n"
    "Relevant entity context:\n{entity_context}\n\n"
    "Community context:\n{community_context}\n\n"
    "Write a precise answer to the query."
)


# ---------------------------------------------------------------------------
# Community Summarizer
# ---------------------------------------------------------------------------


class CommunitySummarizer:
    """
    GraphRAG community summarizer.

    Builds an entity co-occurrence graph, detects communities using
    Louvain clustering, generates LLM summaries for each community,
    then answers queries using global or local search.

    Args:
        llm_fn: Async ``(prompt: str) → str`` callable.
        min_community_size: Minimum entities to form a community.
        max_evidence_chars: Max characters of evidence per community.
        concurrency: Max parallel LLM summarisation calls.

    Example::

        cs = CommunitySummarizer(llm_fn=my_llm)
        cs.add_document("doc1", "Einstein developed the theory of relativity. "
                                "Relativity changed modern physics.", ["Einstein", "relativity", "physics"])
        await cs.build_communities()
        result = await cs.global_query("What are the main scientific themes?")
    """

    def __init__(
        self,
        llm_fn: LLMFn,
        min_community_size: int = 2,
        max_evidence_chars: int = 1000,
        concurrency: int = 5,
    ) -> None:
        self.llm_fn = llm_fn
        self.min_community_size = min_community_size
        self.max_evidence_chars = max_evidence_chars
        self._semaphore = asyncio.Semaphore(concurrency)

        # Entity store
        self._entities: Dict[str, GraphEntity] = {}
        # Co-occurrence: (entity_name_a, entity_name_b) → count
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self._communities: List[Community] = []
        self._built = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        text: str,
        entities: List[str],
        entity_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register a document and its entities in the co-occurrence graph.

        Args:
            doc_id: Unique document identifier.
            text: Document text (used as evidence).
            entities: List of entity name strings found in the document.
            entity_types: Optional ``{entity_name: type}`` mapping.
        """
        types = entity_types or {}
        evidence_snip = text[:300]

        for name in entities:
            key = name.lower()
            if key not in self._entities:
                self._entities[key] = GraphEntity(
                    name=name,
                    entity_type=types.get(name, "unknown"),
                )
            self._entities[key].mentions.append(evidence_snip)

        # Record pairwise co-occurrences (within document)
        unique = list({e.lower() for e in entities})
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = tuple(sorted([unique[i], unique[j]]))
                self._cooccurrence[pair] += 1  # type: ignore[index]

        self._built = False  # invalidate communities

    # ------------------------------------------------------------------
    # Community detection + summarisation
    # ------------------------------------------------------------------

    async def build_communities(self) -> List[Community]:
        """
        Detect communities and generate LLM summaries for each.

        Must be called before :meth:`global_query` or :meth:`local_query`.

        Returns:
            List of :class:`Community` objects with summaries.
        """
        entity_names = list(self._entities.keys())
        n = len(entity_names)
        if n == 0:
            self._communities = []
            self._built = True
            return []

        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=float)
        name_idx = {name: i for i, name in enumerate(entity_names)}
        for (a, b), count in self._cooccurrence.items():
            if a in name_idx and b in name_idx:
                i, j = name_idx[a], name_idx[b]
                adj[i, j] = count
                adj[j, i] = count

        # Louvain clustering
        groups = _louvain_communities(adj)

        # Build Community objects
        raw_communities: List[Community] = []
        for group_idx, node_indices in enumerate(groups):
            if len(node_indices) < self.min_community_size:
                continue
            members = [self._entities[entity_names[i]] for i in node_indices]
            evidence: List[str] = []
            for entity in members:
                evidence.extend(entity.mentions)
            evidence = list(dict.fromkeys(evidence))  # dedup preserving order

            cid = hashlib.md5(
                "|".join(sorted(e.name for e in members)).encode(), usedforsecurity=False
            ).hexdigest()[:8]
            raw_communities.append(
                Community(
                    community_id=cid,
                    entities=members,
                    evidence=evidence[: self.max_evidence_chars // 100],
                    level=0,
                )
            )

        # Generate summaries in parallel
        tasks = [asyncio.create_task(self._summarise_community(c)) for c in raw_communities]
        await asyncio.gather(*tasks, return_exceptions=True)

        self._communities = raw_communities
        self._built = True
        logger.info("CommunitySummarizer: built %d communities", len(self._communities))
        return self._communities

    async def _summarise_community(self, community: Community) -> None:
        """Generate and attach an LLM summary to a community."""
        entities_str = ", ".join(e.name for e in community.entities[:20])
        evidence_str = " ".join(community.evidence)[: self.max_evidence_chars]

        prompt = _COMMUNITY_SUMMARY_PROMPT.format(
            entities=entities_str,
            evidence=evidence_str or "(no evidence)",
        )
        async with self._semaphore:
            try:
                community.summary = await self.llm_fn(prompt)
            except Exception as exc:
                community.summary = f"Community of: {entities_str}"
                logger.warning("CommunitySummarizer: summary failed: %s", exc)

    # ------------------------------------------------------------------
    # Query interfaces
    # ------------------------------------------------------------------

    async def global_query(self, query: str, top_k: int = 5) -> CommunitySummarizerResult:
        """
        Answer a global, corpus-level query using community summaries.

        Best for: "What are the main topics?", "Summarise the corpus."

        Args:
            query: User question.
            top_k: Number of communities to include.

        Returns:
            :class:`CommunitySummarizerResult`.
        """
        if not self._built:
            await self.build_communities()

        # Select communities by summary keyword overlap (simple TF-IDF proxy)
        scored = self._score_communities_for_query(query, top_k)
        summaries_text = "\n\n".join(f"[Community {c.community_id}] {c.summary}" for c in scored)
        prompt = _GLOBAL_ANSWER_PROMPT.format(query=query, summaries=summaries_text)
        answer = await self.llm_fn(prompt)

        return CommunitySummarizerResult(
            query=query,
            answer=answer,
            relevant_communities=scored,
            query_type="global",
        )

    async def local_query(
        self, query: str, entity_name: str, top_k: int = 3
    ) -> CommunitySummarizerResult:
        """
        Answer a local, entity-specific query using graph traversal.

        Best for: "Tell me about Einstein." "What did X do?"

        Args:
            query: User question.
            entity_name: Focal entity name.
            top_k: Number of communities to include.

        Returns:
            :class:`CommunitySummarizerResult`.
        """
        if not self._built:
            await self.build_communities()

        focal_key = entity_name.lower()
        focal_entity = self._entities.get(focal_key)
        entity_context = (
            f"Entity: {focal_entity.name}\n"
            f"Type: {focal_entity.entity_type}\n"
            f"Mentions: {' '.join(focal_entity.mentions[:3])}"
            if focal_entity
            else f"Entity '{entity_name}' not found in graph."
        )

        # Find communities containing this entity
        relevant = [
            c for c in self._communities if any(e.name.lower() == focal_key for e in c.entities)
        ]
        if not relevant:
            relevant = self._score_communities_for_query(query, top_k)
        community_context = "\n\n".join(f"[{c.community_id}] {c.summary}" for c in relevant[:top_k])

        prompt = _LOCAL_ANSWER_PROMPT.format(
            query=query,
            entity_context=entity_context,
            community_context=community_context,
        )
        answer = await self.llm_fn(prompt)

        return CommunitySummarizerResult(
            query=query,
            answer=answer,
            relevant_communities=relevant[:top_k],
            query_type="local",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score_communities_for_query(self, query: str, top_k: int) -> List[Community]:
        """Score communities by keyword overlap with query."""
        q_words = set(query.lower().split())
        scored: List[Tuple[float, Community]] = []
        for c in self._communities:
            text = (c.summary + " " + " ".join(e.name for e in c.entities)).lower()
            overlap = sum(1 for w in q_words if w in text)
            scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    @property
    def community_count(self) -> int:
        return len(self._communities)

    @property
    def entity_count(self) -> int:
        return len(self._entities)
