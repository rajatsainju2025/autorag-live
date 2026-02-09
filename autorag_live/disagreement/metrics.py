from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, cast

import numpy as np
from scipy.stats import kendalltau


@lru_cache(maxsize=256)
def _get_rank_mapping(items_tuple: Tuple[str, ...]) -> dict[str, int]:
    """Cache rank mappings for frequently used lists."""
    return {item: i for i, item in enumerate(items_tuple)}


def jaccard_at_k(list1: List[str], list2: List[str]) -> float:
    """
    Calculates the Jaccard similarity at k between two lists.

    Optimized with early returns and efficient set operations.
    """
    # Early return for empty lists
    if not list1 and not list2:
        return 1.0  # Both empty = identical
    if not list1 or not list2:
        return 0.0  # One empty = no overlap

    set1: Set[str] = set(list1)
    set2: Set[str] = set(list2)

    # Use bitwise operations for efficiency
    intersection = set1 & set2
    union = set1 | set2

    return len(intersection) / len(union)


def kendall_tau_at_k(list1: List[str], list2: List[str]) -> float:
    """
    Calculates Kendall's Tau rank correlation between two lists.
    Optimized with cached rank mappings for frequently used lists.
    """
    # Use cached rank mappings
    rank1 = _get_rank_mapping(tuple(list1))
    rank2 = _get_rank_mapping(tuple(list2))

    # Get all items
    all_items = set(list1) | set(list2)
    all_items_len = len(all_items)

    # Create rank arrays for common items
    ranks1 = [rank1.get(item, all_items_len) for item in all_items]
    ranks2 = [rank2.get(item, all_items_len) for item in all_items]

    tau, _ = kendalltau(ranks1, ranks2)
    return cast(float, tau)


# =============================================================================
# Vectorized metric optimizations using NumPy
# =============================================================================


def batch_jaccard_at_k(lists: List[List[str]]) -> np.ndarray:
    """
    Compute pairwise Jaccard similarity for all list pairs using vectorized ops.

    Returns an (n x n) symmetric similarity matrix.
    Significantly faster than O(n^2) individual jaccard_at_k calls
    for large numbers of retriever result sets.

    Args:
        lists: List of result lists from different retrievers

    Returns:
        np.ndarray of shape (n, n) with pairwise Jaccard similarities
    """
    n = len(lists)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Build universe of all items and encode as binary membership vectors
    universe: Dict[str, int] = {}
    for lst in lists:
        for item in lst:
            if item not in universe:
                universe[item] = len(universe)

    if not universe:
        return np.ones((n, n), dtype=np.float64)

    # Binary membership matrix: (n_lists, n_items)
    membership = np.zeros((n, len(universe)), dtype=np.float32)
    for i, lst in enumerate(lists):
        for item in lst:
            membership[i, universe[item]] = 1.0

    # Vectorized intersection & union via dot products
    # intersection[i,j] = sum(min(a_i, a_j)) = dot product for binary vectors
    intersection = membership @ membership.T
    # union[i,j] = |A_i| + |A_j| - |A_i ∩ A_j|
    sizes = membership.sum(axis=1, keepdims=True)  # (n, 1)
    union = sizes + sizes.T - intersection

    # Avoid division by zero
    result = np.where(union > 0, intersection / union, 1.0)
    return result.astype(np.float64)


def batch_kendall_tau_at_k(lists: List[List[str]]) -> np.ndarray:
    """
    Compute pairwise Kendall-tau for all list pairs using vectorized rank arrays.

    Returns an (n x n) symmetric correlation matrix.

    Args:
        lists: List of ranked result lists from different retrievers

    Returns:
        np.ndarray of shape (n, n) with pairwise Kendall-tau correlations
    """
    n = len(lists)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Build universe
    universe: Dict[str, int] = {}
    for lst in lists:
        for item in lst:
            if item not in universe:
                universe[item] = len(universe)

    if not universe:
        return np.ones((n, n), dtype=np.float64)

    num_items = len(universe)
    default_rank = num_items  # Items not in list get max rank

    # Build rank matrix: (n_lists, n_items)
    rank_matrix = np.full((n, num_items), default_rank, dtype=np.float64)
    for i, lst in enumerate(lists):
        for rank, item in enumerate(lst):
            rank_matrix[i, universe[item]] = rank

    # Compute pairwise Kendall-tau using scipy (vectorized per pair)
    result = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            tau, _ = kendalltau(rank_matrix[i], rank_matrix[j])
            tau_val = float(tau) if not np.isnan(tau) else 0.0
            result[i, j] = tau_val
            result[j, i] = tau_val

    return result


# =============================================================================
# OPTIMIZATION 9: Multi-Model Debate for Improved Answer Quality
# Based on: "Improving Factuality and Reasoning in Language Models through
# Multiagent Debate" (Du et al., 2023) and "Society of Mind" approaches
#
# Implements multi-model debate where multiple LLMs:
# 1. Generate initial responses independently
# 2. Critique each other's responses
# 3. Revise based on critiques
# 4. Vote on final answer or reach consensus
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interactions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...

    @property
    def name(self) -> str:
        """Model identifier."""
        ...


class DebatePhase(str, Enum):
    """Phases of the debate process."""

    INITIAL_RESPONSE = "initial_response"
    CRITIQUE = "critique"
    REVISION = "revision"
    FINAL_VOTE = "final_vote"
    CONSENSUS = "consensus"


@dataclass
class DebateResponse:
    """A response from a model in the debate."""

    model_name: str
    content: str
    phase: DebatePhase
    round_num: int
    confidence: float = 0.0
    sources_cited: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Critique:
    """A critique of another model's response."""

    critic_model: str
    target_model: str
    original_response: str
    critique_text: str
    issues_found: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    agreement_score: float = 0.0  # 0=disagree, 1=agree


@dataclass
class DebateResult:
    """Final result of the debate process."""

    query: str
    final_answer: str
    consensus_reached: bool
    agreement_score: float
    rounds_completed: int
    all_responses: List[DebateResponse] = field(default_factory=list)
    all_critiques: List[Critique] = field(default_factory=list)
    vote_distribution: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiModelDebate:
    """
    Multi-model debate for improved answer quality.

    Orchestrates a debate between multiple LLMs to improve
    factuality and reasoning through iterative refinement.

    Process:
    1. Initial responses: Each model generates independent answer
    2. Critique phase: Models critique each other's responses
    3. Revision phase: Models revise based on critiques
    4. Consensus/Vote: Final answer through agreement or voting

    Example:
        >>> debate = MultiModelDebate([llm1, llm2, llm3])
        >>> result = await debate.run_debate(
        ...     query="What causes climate change?",
        ...     context=["Source 1...", "Source 2..."],
        ...     max_rounds=3
        ... )
        >>> print(f"Final answer: {result.final_answer}")
        >>> print(f"Agreement: {result.agreement_score:.0%}")
    """

    def __init__(
        self,
        models: List[LLMProtocol],
        consensus_threshold: float = 0.8,
        max_rounds: int = 3,
        temperature: float = 0.7,
    ):
        """
        Initialize multi-model debate.

        Args:
            models: List of LLM models to participate
            consensus_threshold: Agreement level for consensus
            max_rounds: Maximum debate rounds
            temperature: Generation temperature
        """
        self.models = models
        self.consensus_threshold = consensus_threshold
        self.max_rounds = max_rounds
        self.temperature = temperature

        # Statistics
        self._stats = {
            "debates_run": 0,
            "consensus_reached": 0,
            "avg_rounds": 0.0,
            "avg_agreement": 0.0,
        }

    async def run_debate(
        self,
        query: str,
        context: Optional[List[str]] = None,
        max_rounds: Optional[int] = None,
    ) -> DebateResult:
        """
        Run a multi-model debate.

        Args:
            query: User query
            context: Retrieved context passages
            max_rounds: Override max rounds

        Returns:
            DebateResult with final answer and metadata
        """
        max_rounds = max_rounds or self.max_rounds

        all_responses: List[DebateResponse] = []
        all_critiques: List[Critique] = []

        # Phase 1: Initial responses
        initial_responses = await self._get_initial_responses(query, context)
        all_responses.extend(initial_responses)

        # Check early consensus
        agreement = self._compute_agreement(initial_responses)
        if agreement >= self.consensus_threshold:
            return self._create_result(
                query,
                initial_responses,
                all_critiques,
                1,
                True,
                agreement,
            )

        # Iterative debate rounds
        current_responses = initial_responses
        for round_num in range(2, max_rounds + 1):
            # Phase 2: Critique
            critiques = await self._generate_critiques(query, context, current_responses)
            all_critiques.extend(critiques)

            # Phase 3: Revision
            revised_responses = await self._generate_revisions(
                query, context, current_responses, critiques, round_num
            )
            all_responses.extend(revised_responses)
            current_responses = revised_responses

            # Check consensus
            agreement = self._compute_agreement(current_responses)
            if agreement >= self.consensus_threshold:
                self._stats["consensus_reached"] += 1
                return self._create_result(
                    query,
                    all_responses,
                    all_critiques,
                    round_num,
                    True,
                    agreement,
                )

        # Phase 4: Final vote if no consensus
        final_answer, vote_dist = await self._final_vote(query, context, current_responses)

        self._stats["debates_run"] += 1

        return DebateResult(
            query=query,
            final_answer=final_answer,
            consensus_reached=False,
            agreement_score=agreement,
            rounds_completed=max_rounds,
            all_responses=all_responses,
            all_critiques=all_critiques,
            vote_distribution=vote_dist,
        )

    async def _get_initial_responses(
        self,
        query: str,
        context: Optional[List[str]] = None,
    ) -> List[DebateResponse]:
        """Get initial responses from all models."""
        context_text = "\n\n".join(context) if context else "No context provided."

        prompt = f"""Answer this question based on the provided context.

Context:
{context_text}

Question: {query}

Provide a clear, accurate answer. Cite sources when making claims.

Answer:"""

        responses = []
        for model in self.models:
            try:
                content = await model.generate(prompt, temperature=self.temperature)
                responses.append(
                    DebateResponse(
                        model_name=getattr(model, "name", "unknown"),
                        content=content.strip(),
                        phase=DebatePhase.INITIAL_RESPONSE,
                        round_num=1,
                    )
                )
            except Exception:
                # Skip failed models
                pass

        return responses

    async def _generate_critiques(
        self,
        query: str,
        context: Optional[List[str]],
        responses: List[DebateResponse],
    ) -> List[Critique]:
        """Generate critiques between models."""
        critiques = []

        for i, critic_response in enumerate(responses):
            for j, target_response in enumerate(responses):
                if i == j:
                    continue  # Don't self-critique

                # Find critic model
                critic_model = self.models[i % len(self.models)]

                critique_prompt = f"""You are reviewing another model's answer.

Question: {query}

Their answer:
{target_response.content}

Evaluate this answer for:
1. Factual accuracy (based on provided context)
2. Logical consistency
3. Completeness
4. Clarity

Provide specific issues found and suggested improvements.

Format your response:
ISSUES:
- [list issues]

IMPROVEMENTS:
- [list improvements]

AGREEMENT_SCORE: [0.0-1.0]

Review:"""

                try:
                    critique_text = await critic_model.generate(critique_prompt, temperature=0.3)

                    # Parse critique
                    issues = self._extract_list(critique_text, "ISSUES")
                    improvements = self._extract_list(critique_text, "IMPROVEMENTS")
                    agreement = self._extract_score(critique_text)

                    critiques.append(
                        Critique(
                            critic_model=getattr(critic_model, "name", f"model_{i}"),
                            target_model=target_response.model_name,
                            original_response=target_response.content,
                            critique_text=critique_text,
                            issues_found=issues,
                            suggested_improvements=improvements,
                            agreement_score=agreement,
                        )
                    )
                except Exception:
                    pass

        return critiques

    async def _generate_revisions(
        self,
        query: str,
        context: Optional[List[str]],
        responses: List[DebateResponse],
        critiques: List[Critique],
        round_num: int,
    ) -> List[DebateResponse]:
        """Generate revised responses based on critiques."""
        revised = []

        for i, (model, response) in enumerate(zip(self.models, responses)):
            # Gather critiques for this model
            relevant_critiques = [c for c in critiques if c.target_model == response.model_name]

            critique_summary = "\n".join(
                f"Critic {c.critic_model}: {c.critique_text[:200]}..." for c in relevant_critiques
            )

            context_text = "\n\n".join(context) if context else "No context provided."

            revision_prompt = f"""Revise your answer based on peer feedback.

Question: {query}

Context:
{context_text}

Your previous answer:
{response.content}

Feedback from other models:
{critique_summary}

Consider the feedback and revise your answer. Maintain accuracy while
addressing valid criticisms.

Revised answer:"""

            try:
                content = await model.generate(revision_prompt, temperature=0.5)
                revised.append(
                    DebateResponse(
                        model_name=getattr(model, "name", f"model_{i}"),
                        content=content.strip(),
                        phase=DebatePhase.REVISION,
                        round_num=round_num,
                    )
                )
            except Exception:
                # Keep original if revision fails
                revised.append(response)

        return revised

    async def _final_vote(
        self,
        query: str,
        context: Optional[List[str]],
        responses: List[DebateResponse],
    ) -> Tuple[str, Dict[str, int]]:
        """Final voting to select best answer."""
        vote_counts: Dict[str, int] = {}

        # Each model votes for best answer (not their own)
        for i, voter_model in enumerate(self.models):
            other_responses = [r for j, r in enumerate(responses) if j != i]

            if not other_responses:
                continue

            options = "\n\n".join(
                f"Option {j+1} ({r.model_name}):\n{r.content}"
                for j, r in enumerate(other_responses)
            )

            vote_prompt = f"""Choose the best answer to this question.

Question: {query}

{options}

Which option is most accurate and complete? Respond with just the option number.

Best option:"""

            try:
                vote = await voter_model.generate(vote_prompt, temperature=0.1)
                # Parse vote
                vote_num = int("".join(c for c in vote if c.isdigit()) or "1")
                if 1 <= vote_num <= len(other_responses):
                    voted_model = other_responses[vote_num - 1].model_name
                    vote_counts[voted_model] = vote_counts.get(voted_model, 0) + 1
            except Exception:
                pass

        # Select winner
        if vote_counts:
            winner = max(vote_counts.items(), key=lambda x: x[1])[0]
            winning_response = next(r.content for r in responses if r.model_name == winner)
            return winning_response, vote_counts

        # Fallback to first response
        return responses[0].content if responses else "", vote_counts

    def _compute_agreement(self, responses: List[DebateResponse]) -> float:
        """Compute agreement level between responses."""
        if len(responses) < 2:
            return 1.0

        # Simple word overlap agreement
        word_sets = [set(r.content.lower().split()) for r in responses]

        total_overlap = 0.0
        comparisons = 0

        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = word_sets[i] & word_sets[j]
                union = word_sets[i] | word_sets[j]
                if union:
                    total_overlap += len(intersection) / len(union)
                comparisons += 1

        return total_overlap / max(1, comparisons)

    def _create_result(
        self,
        query: str,
        responses: List[DebateResponse],
        critiques: List[Critique],
        rounds: int,
        consensus: bool,
        agreement: float,
    ) -> DebateResult:
        """Create debate result from consensus."""
        # Use last round's responses for final answer
        final_responses = [r for r in responses if r.round_num == rounds]
        if not final_responses:
            final_responses = responses

        # Combine responses for final answer
        final_answer = final_responses[0].content if final_responses else ""

        return DebateResult(
            query=query,
            final_answer=final_answer,
            consensus_reached=consensus,
            agreement_score=agreement,
            rounds_completed=rounds,
            all_responses=responses,
            all_critiques=critiques,
        )

    def _extract_list(self, text: str, section: str) -> List[str]:
        """Extract list items from a section."""
        import re

        pattern = rf"{section}:\s*\n((?:\s*-[^\n]+\n?)+)"
        match = re.search(pattern, text, re.I)
        if match:
            items = re.findall(r"-\s*(.+)", match.group(1))
            return items
        return []

    def _extract_score(self, text: str) -> float:
        """Extract agreement score from text."""
        import re

        match = re.search(r"AGREEMENT_SCORE:\s*([0-9.]+)", text, re.I)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5

    def get_stats(self) -> Dict[str, Any]:
        """Get debate statistics."""
        return dict(self._stats)


class ChainOfDebate:
    """
    Chain-of-debate pattern for sequential refinement.

    Models take turns building on previous responses
    rather than parallel debate.
    """

    def __init__(
        self,
        models: List[LLMProtocol],
        refinement_rounds: int = 2,
    ):
        """Initialize chain of debate."""
        self.models = models
        self.refinement_rounds = refinement_rounds

    async def run_chain(
        self,
        query: str,
        context: Optional[List[str]] = None,
    ) -> DebateResult:
        """Run sequential chain of debate."""
        all_responses: List[DebateResponse] = []
        current_answer = ""

        for round_num in range(self.refinement_rounds):
            for i, model in enumerate(self.models):
                context_text = "\n".join(context) if context else "No context."

                if current_answer:
                    prompt = f"""Build upon and improve this answer.

Question: {query}

Context:
{context_text}

Previous answer:
{current_answer}

Improve this answer by:
1. Adding missing information
2. Correcting any errors
3. Clarifying unclear points

Improved answer:"""
                else:
                    prompt = f"""Answer this question.

Question: {query}

Context:
{context_text}

Answer:"""

                try:
                    response = await model.generate(prompt, temperature=0.5)
                    current_answer = response.strip()

                    all_responses.append(
                        DebateResponse(
                            model_name=getattr(model, "name", f"model_{i}"),
                            content=current_answer,
                            phase=DebatePhase.REVISION
                            if round_num > 0
                            else DebatePhase.INITIAL_RESPONSE,
                            round_num=round_num + 1,
                        )
                    )
                except Exception:
                    pass

        return DebateResult(
            query=query,
            final_answer=current_answer,
            consensus_reached=True,  # Chain implies consensus
            agreement_score=1.0,
            rounds_completed=self.refinement_rounds,
            all_responses=all_responses,
            all_critiques=[],
        )


def create_debate_ensemble(
    models: List[LLMProtocol],
    consensus_threshold: float = 0.8,
) -> MultiModelDebate:
    """
    Create a multi-model debate ensemble.

    Args:
        models: LLM models for debate
        consensus_threshold: Threshold for consensus

    Returns:
        Configured MultiModelDebate
    """
    return MultiModelDebate(
        models=models,
        consensus_threshold=consensus_threshold,
    )
