"""
Multi-Hop Reasoning Engine Module.

Implements iterative retrieval with chain-of-thought reasoning,
following the IRCoT (Trivedi et al., 2023) pattern.

Key Features:
1. Iterative Retrieval-augmented CoT reasoning
2. Evidence chain construction
3. Sub-question decomposition
4. Answer aggregation from multiple hops
5. Confidence-based termination

Example:
    >>> engine = MultiHopEngine(llm, retriever)
    >>> result = await engine.reason("Who founded the company that made the iPhone?")
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from autorag_live.core.protocols import BaseLLM, Message

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class HopType(str, Enum):
    """Types of reasoning hops."""

    DECOMPOSE = "decompose"  # Break down query
    RETRIEVE = "retrieve"  # Fetch evidence
    REASON = "reason"  # Chain-of-thought
    BRIDGE = "bridge"  # Connect evidence
    AGGREGATE = "aggregate"  # Combine findings


@dataclass
class Evidence:
    """
    Evidence collected during reasoning.

    Attributes:
        content: Evidence text
        source_id: Source document ID
        relevance_score: How relevant to the query
        hop_number: Which hop retrieved this
        question: Sub-question this answers
    """

    content: str
    source_id: str = ""
    relevance_score: float = 0.8
    hop_number: int = 0
    question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source_id": self.source_id,
            "relevance_score": self.relevance_score,
            "hop_number": self.hop_number,
            "question": self.question,
        }


@dataclass
class ReasoningHop:
    """
    A single hop in the reasoning chain.

    Attributes:
        hop_number: Sequential hop number
        hop_type: Type of reasoning step
        input_query: Query for this hop
        thought: Chain-of-thought reasoning
        action: Action taken
        observation: Result of action
        evidence: Evidence collected
        sub_answer: Partial answer from this hop
    """

    hop_number: int
    hop_type: HopType
    input_query: str
    thought: str = ""
    action: str = ""
    observation: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    sub_answer: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hop_number": self.hop_number,
            "hop_type": self.hop_type.value,
            "input_query": self.input_query,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "evidence": [e.to_dict() for e in self.evidence],
            "sub_answer": self.sub_answer,
        }


@dataclass
class MultiHopResult:
    """
    Result from multi-hop reasoning.

    Attributes:
        answer: Final synthesized answer
        reasoning_chain: List of reasoning hops
        total_hops: Number of hops taken
        evidence_chain: All evidence collected
        confidence: Answer confidence
        latency_ms: Total execution time
    """

    answer: str
    reasoning_chain: List[ReasoningHop] = field(default_factory=list)
    total_hops: int = 0
    evidence_chain: List[Evidence] = field(default_factory=list)
    confidence: float = 0.8
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "reasoning_chain": [h.to_dict() for h in self.reasoning_chain],
            "total_hops": self.total_hops,
            "evidence_count": len(self.evidence_chain),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }


# =============================================================================
# Query Decomposer
# =============================================================================


class QueryDecomposer:
    """
    Decomposes complex queries into sub-questions.

    Handles multi-hop queries by identifying intermediate questions.
    """

    # Patterns indicating multi-hop
    BRIDGE_PATTERNS = [
        r"(\w+)\s+(?:that|which|who)\s+(\w+)",  # "X that Y"
        r"(\w+)'s\s+(\w+)",  # "X's Y"
        r"founder\s+of\s+(\w+)",  # Entity relationships
        r"(?:CEO|president|founder)\s+of\s+",
        r"company\s+(?:that|which)\s+(?:made|created|built)",
    ]

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize decomposer."""
        self.llm = llm

    def detect_multi_hop(self, query: str) -> bool:
        """Check if query requires multi-hop reasoning."""
        query_lower = query.lower()

        # Check for bridge patterns
        for pattern in self.BRIDGE_PATTERNS:
            if re.search(pattern, query_lower):
                return True

        # Check for multiple entities
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        if len(entities) > 2:
            return True

        # Check for nested questions
        if query.count("?") > 1:
            return True

        return False

    def decompose_heuristic(self, query: str) -> List[str]:
        """
        Decompose using heuristic rules.

        Args:
            query: Complex query

        Returns:
            List of sub-questions
        """
        sub_questions = []
        query_lower = query.lower()

        # Check for "X that Y" pattern
        match = re.search(r"(.+?)\s+(?:that|which)\s+(.+)", query)
        if match:
            # Entity reference pattern
            entity_part = match.group(1)
            action_part = match.group(2)

            # Create bridging questions
            sub_questions.append(f"What {action_part}?")
            sub_questions.append(f"What is {entity_part} of the answer to the previous question?")
            return sub_questions

        # Check for possessive pattern (X's Y)
        match = re.search(r"(.+?)'s\s+(.+)", query)
        if match:
            owner = match.group(1)
            attribute = match.group(2)
            sub_questions.append(f"Who or what is {owner}?")
            sub_questions.append(f"What is their/its {attribute}?")
            return sub_questions

        # Check for relationship patterns
        if "founder of" in query_lower or "ceo of" in query_lower:
            # Extract the entity
            match = re.search(r"(?:founder|ceo)\s+of\s+(.+?)(?:\?|$)", query_lower)
            if match:
                entity = match.group(1)
                sub_questions.append(f"What is {entity}?")
                sub_questions.append(f"Who founded/leads {entity}?")
                return sub_questions

        # Default: return original
        return [query]

    async def decompose_with_llm(self, query: str) -> List[str]:
        """
        Decompose using LLM.

        Args:
            query: Complex query

        Returns:
            List of sub-questions
        """
        if not self.llm:
            return self.decompose_heuristic(query)

        prompt = f"""Break down this complex question into simpler sub-questions that can be answered step-by-step.

Complex Question: {query}

Rules:
- Each sub-question should be answerable independently
- Order them logically (answer earlier ones first)
- The final sub-question should lead to the original answer
- Return 2-4 sub-questions

Format each on a new line starting with a number:
1. <first sub-question>
2. <second sub-question>
..."""

        try:
            result = await self.llm.generate(
                [Message.user(prompt)],
                temperature=0.3,
                max_tokens=300,
            )

            sub_questions = []
            for line in result.content.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove number prefix
                    cleaned = re.sub(r"^\d+[\.\):\-]\s*", "", line)
                    if cleaned:
                        sub_questions.append(cleaned)

            return sub_questions if sub_questions else [query]

        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")
            return self.decompose_heuristic(query)


# =============================================================================
# Chain of Thought Reasoner
# =============================================================================


class ChainOfThoughtReasoner:
    """
    Generates chain-of-thought reasoning interleaved with retrieval.

    Implements the IRCoT pattern.
    """

    def __init__(self, llm: BaseLLM):
        """Initialize reasoner."""
        self.llm = llm

    async def generate_thought(
        self,
        query: str,
        context: str,
        previous_thoughts: List[str],
    ) -> str:
        """
        Generate chain-of-thought reasoning.

        Args:
            query: Current question
            context: Retrieved context
            previous_thoughts: Previous reasoning steps

        Returns:
            New thought/reasoning
        """
        prev_chain = "\n".join(f"- {t}" for t in previous_thoughts[-3:])

        prompt = f"""Given the question and context, reason step-by-step towards an answer.

Question: {query}

Context:
{context}

Previous reasoning:
{prev_chain if prev_chain else "None yet"}

Think step-by-step:
1. What key information is in the context?
2. How does it relate to the question?
3. What can we conclude?

Reasoning:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.3,
            max_tokens=300,
        )

        return result.content.strip()

    async def should_continue(
        self,
        query: str,
        current_answer: str,
        evidence: List[Evidence],
    ) -> Tuple[bool, str]:
        """
        Decide if more reasoning hops are needed.

        Returns:
            (should_continue, reason)
        """
        evidence_summary = "\n".join(e.content[:200] for e in evidence[-5:])

        prompt = f"""Determine if we have enough information to answer the question.

Question: {query}

Current answer: {current_answer}

Evidence collected:
{evidence_summary}

Is the answer complete and well-supported? Respond:
- "COMPLETE: <reason>" if we can confidently answer
- "NEED_MORE: <what's missing>" if more information is needed

Response:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.0,
            max_tokens=100,
        )

        response = result.content.strip()
        if response.startswith("COMPLETE"):
            return False, response
        return True, response

    async def synthesize_answer(
        self,
        query: str,
        evidence: List[Evidence],
        reasoning_chain: List[str],
    ) -> str:
        """
        Synthesize final answer from evidence and reasoning.

        Args:
            query: Original query
            evidence: All collected evidence
            reasoning_chain: All reasoning steps

        Returns:
            Final synthesized answer
        """
        evidence_text = "\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidence[:10]))
        reasoning_text = "\n".join(f"- {r}" for r in reasoning_chain[-5:])

        prompt = f"""Synthesize a comprehensive answer based on the evidence and reasoning.

Original Question: {query}

Evidence:
{evidence_text}

Reasoning Chain:
{reasoning_text}

Provide a clear, well-supported answer:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.5,
            max_tokens=500,
        )

        return result.content.strip()


# =============================================================================
# Multi-Hop Engine
# =============================================================================


class MultiHopEngine:
    """
    Multi-hop reasoning engine with iterative retrieval.

    Implements IRCoT (Interleaved Retrieval Chain-of-Thought) pattern
    for complex queries requiring multiple reasoning steps.

    Example:
        >>> engine = MultiHopEngine(llm, retriever)
        >>> result = await engine.reason("Who founded Apple's main competitor?")
        >>> print(result.answer)
    """

    MAX_HOPS = 5
    DEFAULT_K = 3

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Optional[Callable] = None,
        *,
        max_hops: int = 5,
        k_per_hop: int = 3,
        use_llm_decomposition: bool = True,
    ):
        """
        Initialize engine.

        Args:
            llm: Language model
            retriever: Retriever function
            max_hops: Maximum reasoning hops
            k_per_hop: Documents to retrieve per hop
            use_llm_decomposition: Use LLM for query decomposition
        """
        self.llm = llm
        self.retriever = retriever
        self.max_hops = min(max_hops, self.MAX_HOPS)
        self.k_per_hop = k_per_hop

        self.decomposer = QueryDecomposer(llm if use_llm_decomposition else None)
        self.reasoner = ChainOfThoughtReasoner(llm)

    async def reason(
        self,
        query: str,
        *,
        max_hops: Optional[int] = None,
    ) -> MultiHopResult:
        """
        Perform multi-hop reasoning.

        Args:
            query: User query
            max_hops: Override max hops

        Returns:
            MultiHopResult with answer and reasoning chain
        """
        start_time = time.time()
        max_h = max_hops or self.max_hops

        # Check if multi-hop is needed
        needs_multi_hop = self.decomposer.detect_multi_hop(query)

        if not needs_multi_hop:
            # Single-hop reasoning
            return await self._single_hop_reason(query)

        # Multi-hop reasoning
        reasoning_chain: List[ReasoningHop] = []
        evidence_chain: List[Evidence] = []
        thoughts: List[str] = []
        current_answer = ""

        # Decompose query
        if self.decomposer.llm:
            sub_questions = await self.decomposer.decompose_with_llm(query)
        else:
            sub_questions = self.decomposer.decompose_heuristic(query)

        # Execute hops
        for hop_num in range(max_h):
            hop_start = time.time()

            # Determine current sub-question
            if hop_num < len(sub_questions):
                current_query = sub_questions[hop_num]
            else:
                current_query = query

            # Retrieve evidence
            hop_evidence = await self._retrieve(current_query, hop_num)
            evidence_chain.extend(hop_evidence)

            # Reason over evidence
            context = "\n".join(e.content for e in hop_evidence)
            thought = await self.reasoner.generate_thought(
                current_query,
                context,
                thoughts,
            )
            thoughts.append(thought)

            # Generate sub-answer
            sub_answer = await self._generate_sub_answer(
                current_query,
                hop_evidence,
                thought,
            )

            # Update current answer
            if sub_answer:
                current_answer = sub_answer

            # Create hop record
            hop = ReasoningHop(
                hop_number=hop_num,
                hop_type=HopType.RETRIEVE
                if hop_num < len(sub_questions) - 1
                else HopType.AGGREGATE,
                input_query=current_query,
                thought=thought,
                action=f"Retrieved {len(hop_evidence)} documents",
                observation=context[:500],
                evidence=hop_evidence,
                sub_answer=sub_answer,
                latency_ms=(time.time() - hop_start) * 1000,
            )
            reasoning_chain.append(hop)

            # Check if we should continue
            should_continue, reason = await self.reasoner.should_continue(
                query,
                current_answer,
                evidence_chain,
            )

            if not should_continue:
                break

        # Synthesize final answer
        final_answer = await self.reasoner.synthesize_answer(
            query,
            evidence_chain,
            thoughts,
        )

        return MultiHopResult(
            answer=final_answer,
            reasoning_chain=reasoning_chain,
            total_hops=len(reasoning_chain),
            evidence_chain=evidence_chain,
            confidence=self._compute_confidence(reasoning_chain),
            latency_ms=(time.time() - start_time) * 1000,
            metadata={
                "sub_questions": sub_questions,
                "needs_multi_hop": needs_multi_hop,
            },
        )

    async def _single_hop_reason(self, query: str) -> MultiHopResult:
        """Handle single-hop queries."""
        start_time = time.time()

        # Retrieve
        evidence = await self._retrieve(query, 0)

        # Generate answer
        context = "\n".join(e.content for e in evidence)

        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {query}

Answer:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.7,
        )

        hop = ReasoningHop(
            hop_number=0,
            hop_type=HopType.RETRIEVE,
            input_query=query,
            thought="Single-hop retrieval",
            action=f"Retrieved {len(evidence)} documents",
            observation=context[:500],
            evidence=evidence,
            sub_answer=result.content,
        )

        return MultiHopResult(
            answer=result.content,
            reasoning_chain=[hop],
            total_hops=1,
            evidence_chain=evidence,
            confidence=0.8,
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _retrieve(
        self,
        query: str,
        hop_number: int,
    ) -> List[Evidence]:
        """Retrieve evidence for a hop."""
        if not self.retriever:
            return []

        try:
            if asyncio.iscoroutinefunction(self.retriever):
                result = await self.retriever(query, self.k_per_hop)
            else:
                result = self.retriever(query, self.k_per_hop)

            docs = result if isinstance(result, list) else result.documents

            return [
                Evidence(
                    content=doc.content,
                    source_id=doc.id,
                    relevance_score=doc.score,
                    hop_number=hop_number,
                    question=query,
                )
                for doc in docs
            ]
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

    async def _generate_sub_answer(
        self,
        query: str,
        evidence: List[Evidence],
        thought: str,
    ) -> str:
        """Generate answer for a sub-question."""
        context = "\n".join(e.content for e in evidence)

        prompt = f"""Based on the context and reasoning, answer the question concisely.

Question: {query}

Context:
{context}

Reasoning: {thought}

Concise answer:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.3,
            max_tokens=200,
        )

        return result.content.strip()

    def _compute_confidence(
        self,
        reasoning_chain: List[ReasoningHop],
    ) -> float:
        """Compute confidence from reasoning chain."""
        if not reasoning_chain:
            return 0.5

        # Average evidence relevance
        all_evidence = [e for hop in reasoning_chain for e in hop.evidence]
        if all_evidence:
            avg_relevance = sum(e.relevance_score for e in all_evidence) / len(all_evidence)
        else:
            avg_relevance = 0.5

        # Penalize too many hops
        hop_penalty = max(0, 1 - len(reasoning_chain) * 0.1)

        return avg_relevance * 0.7 + hop_penalty * 0.3


# =============================================================================
# Convenience Functions
# =============================================================================


def create_multi_hop_engine(
    llm: BaseLLM,
    retriever: Optional[Callable] = None,
    max_hops: int = 5,
) -> MultiHopEngine:
    """
    Create a multi-hop reasoning engine.

    Args:
        llm: Language model
        retriever: Retriever function
        max_hops: Maximum hops

    Returns:
        MultiHopEngine
    """
    return MultiHopEngine(
        llm=llm,
        retriever=retriever,
        max_hops=max_hops,
    )
