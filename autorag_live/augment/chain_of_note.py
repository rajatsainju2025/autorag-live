"""
Chain-of-Note (CoN) Implementation.

Based on: "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models"
(Yu et al., 2023)

Chain-of-Note improves RAG robustness by:
1. Reading each retrieved document and generating a structured note
2. Evaluating document relevance through note quality
3. Aggregating notes to form comprehensive evidence
4. Generating final answer from aggregated notes

This helps handle noisy retrieval and improve answer quality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols and Types
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval backends."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list["RetrievedDocument"]:
        """Retrieve documents for query."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


# ============================================================================
# Note Types and Structures
# ============================================================================


class NoteType(Enum):
    """Types of notes that can be generated."""

    RELEVANT = "relevant"  # Document directly addresses query
    PARTIALLY_RELEVANT = "partially_relevant"  # Some useful info
    IRRELEVANT = "irrelevant"  # Document not relevant
    CONTRADICTORY = "contradictory"  # Document contradicts other evidence
    UNKNOWN = "unknown"  # Cannot determine relevance


class EvidenceStrength(Enum):
    """Strength of evidence from a document."""

    STRONG = "strong"  # Direct, authoritative evidence
    MODERATE = "moderate"  # Supportive but indirect
    WEAK = "weak"  # Tangentially related
    NONE = "none"  # No relevant evidence


@dataclass
class Note:
    """A structured note about a document."""

    document: RetrievedDocument
    note_type: NoteType
    summary: str
    key_facts: list[str]
    relevance_score: float  # 0.0 to 1.0
    evidence_strength: EvidenceStrength
    reasoning: str
    contradictions: list[str] = field(default_factory=list)


@dataclass
class AggregatedNotes:
    """Aggregated notes from multiple documents."""

    notes: list[Note]
    consensus_facts: list[str]
    conflicting_facts: list[str]
    evidence_gaps: list[str]
    overall_confidence: float


@dataclass
class ChainOfNoteResult:
    """Result from Chain-of-Note processing."""

    query: str
    notes: list[Note]
    aggregated: AggregatedNotes
    final_answer: str
    reasoning_chain: list[str]
    confidence: float


# ============================================================================
# Note Generation
# ============================================================================


class NoteGenerator(ABC):
    """Abstract base for note generation."""

    @abstractmethod
    async def generate_note(
        self,
        query: str,
        document: RetrievedDocument,
    ) -> Note:
        """Generate a structured note for a document."""
        ...


class LLMNoteGenerator(NoteGenerator):
    """LLM-based note generator."""

    NOTE_TEMPLATE = """You are analyzing a document to understand its relevance to a query.
Read the document carefully and generate a structured note.

Query: {query}

Document:
{document}

Generate a note with the following structure:

1. Summary (2-3 sentences):
What are the main points of this document?

2. Relevance Assessment:
How relevant is this document to the query? (RELEVANT/PARTIALLY_RELEVANT/IRRELEVANT)

3. Key Facts:
List 3-5 specific facts from the document that relate to the query.
If none exist, write "No relevant facts found."

4. Evidence Strength (STRONG/MODERATE/WEAK/NONE):
How strong is the evidence this document provides?

5. Reasoning:
Explain why you assessed the relevance and evidence strength as you did.

Note:"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def generate_note(
        self,
        query: str,
        document: RetrievedDocument,
    ) -> Note:
        """Generate a structured note using LLM."""
        prompt = self.NOTE_TEMPLATE.format(
            query=query,
            document=document.content[:2000],  # Truncate long documents
        )

        response = await self.llm.generate(prompt, max_tokens=500, temperature=0.3)

        return self._parse_note(document, response)

    def _parse_note(
        self,
        document: RetrievedDocument,
        response: str,
    ) -> Note:
        """Parse LLM response into structured note."""
        # Extract sections from response
        lines = response.strip().split("\n")

        summary = ""
        note_type = NoteType.UNKNOWN
        key_facts: list[str] = []
        evidence_strength = EvidenceStrength.NONE
        reasoning = ""

        current_section = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if "Summary" in line or "summary" in line:
                current_section = "summary"
            elif "Relevance" in line or "relevance" in line:
                current_section = "relevance"
            elif "Key Facts" in line or "key facts" in line:
                current_section = "facts"
            elif "Evidence Strength" in line or "evidence" in line:
                current_section = "evidence"
            elif "Reasoning" in line or "reasoning" in line:
                current_section = "reasoning"
            else:
                # Add content to current section
                if current_section == "summary":
                    summary += " " + line
                elif current_section == "relevance":
                    note_type = self._parse_relevance(line)
                elif current_section == "facts":
                    if line.startswith("-") or line.startswith("•"):
                        key_facts.append(line[1:].strip())
                    elif line[0].isdigit():
                        key_facts.append(line.split(".", 1)[-1].strip())
                elif current_section == "evidence":
                    evidence_strength = self._parse_evidence_strength(line)
                elif current_section == "reasoning":
                    reasoning += " " + line

        # Calculate relevance score based on note type
        relevance_scores = {
            NoteType.RELEVANT: 1.0,
            NoteType.PARTIALLY_RELEVANT: 0.6,
            NoteType.IRRELEVANT: 0.1,
            NoteType.CONTRADICTORY: 0.3,
            NoteType.UNKNOWN: 0.5,
        }

        return Note(
            document=document,
            note_type=note_type,
            summary=summary.strip(),
            key_facts=key_facts[:5],
            relevance_score=relevance_scores.get(note_type, 0.5),
            evidence_strength=evidence_strength,
            reasoning=reasoning.strip(),
        )

    def _parse_relevance(self, text: str) -> NoteType:
        """Parse relevance from text."""
        text_upper = text.upper()
        if "PARTIALLY" in text_upper:
            return NoteType.PARTIALLY_RELEVANT
        elif "IRRELEVANT" in text_upper:
            return NoteType.IRRELEVANT
        elif "RELEVANT" in text_upper:
            return NoteType.RELEVANT
        elif "CONTRADICTORY" in text_upper:
            return NoteType.CONTRADICTORY
        return NoteType.UNKNOWN

    def _parse_evidence_strength(self, text: str) -> EvidenceStrength:
        """Parse evidence strength from text."""
        text_upper = text.upper()
        if "STRONG" in text_upper:
            return EvidenceStrength.STRONG
        elif "MODERATE" in text_upper:
            return EvidenceStrength.MODERATE
        elif "WEAK" in text_upper:
            return EvidenceStrength.WEAK
        return EvidenceStrength.NONE


# ============================================================================
# Note Aggregation
# ============================================================================


class NoteAggregator(ABC):
    """Abstract base for note aggregation."""

    @abstractmethod
    async def aggregate(
        self,
        query: str,
        notes: list[Note],
    ) -> AggregatedNotes:
        """Aggregate multiple notes into unified evidence."""
        ...


class LLMNoteAggregator(NoteAggregator):
    """LLM-based note aggregation."""

    AGGREGATION_TEMPLATE = """You are aggregating notes from multiple documents to answer a query.

Query: {query}

Notes from Documents:
{formatted_notes}

Based on these notes:

1. Consensus Facts:
What facts are supported by multiple documents?

2. Conflicting Information:
What information conflicts between documents?

3. Evidence Gaps:
What information is missing that would help answer the query?

4. Overall Confidence:
On a scale of 0-100, how confident should we be in answering this query based on these notes?

Aggregation:"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def aggregate(
        self,
        query: str,
        notes: list[Note],
    ) -> AggregatedNotes:
        """Aggregate notes using LLM."""
        formatted_notes = self._format_notes(notes)

        prompt = self.AGGREGATION_TEMPLATE.format(
            query=query,
            formatted_notes=formatted_notes,
        )

        response = await self.llm.generate(prompt, max_tokens=600, temperature=0.3)

        return self._parse_aggregation(notes, response)

    def _format_notes(self, notes: list[Note]) -> str:
        """Format notes for prompt."""
        formatted = []
        for i, note in enumerate(notes, 1):
            if note.note_type != NoteType.IRRELEVANT:
                note_text = f"""
Note {i} ({note.note_type.value}, Evidence: {note.evidence_strength.value}):
Summary: {note.summary}
Key Facts: {', '.join(note.key_facts) if note.key_facts else 'None'}
"""
                formatted.append(note_text)

        return "\n".join(formatted)

    def _parse_aggregation(
        self,
        notes: list[Note],
        response: str,
    ) -> AggregatedNotes:
        """Parse aggregation response."""
        lines = response.strip().split("\n")

        consensus_facts: list[str] = []
        conflicting_facts: list[str] = []
        evidence_gaps: list[str] = []
        confidence = 0.5

        current_section = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect sections
            if "Consensus" in line or "consensus" in line:
                current_section = "consensus"
            elif "Conflict" in line or "conflict" in line:
                current_section = "conflicts"
            elif "Gap" in line or "gap" in line or "Missing" in line:
                current_section = "gaps"
            elif "Confidence" in line or "confidence" in line:
                current_section = "confidence"
            else:
                # Parse content
                if current_section == "consensus":
                    if line.startswith("-") or line[0].isdigit():
                        consensus_facts.append(line.lstrip("-•").split(".", 1)[-1].strip())
                elif current_section == "conflicts":
                    if line.startswith("-") or line[0].isdigit():
                        conflicting_facts.append(line.lstrip("-•").split(".", 1)[-1].strip())
                elif current_section == "gaps":
                    if line.startswith("-") or line[0].isdigit():
                        evidence_gaps.append(line.lstrip("-•").split(".", 1)[-1].strip())
                elif current_section == "confidence":
                    # Extract number from line
                    import re

                    numbers = re.findall(r"\d+", line)
                    if numbers:
                        confidence = float(numbers[0]) / 100.0

        return AggregatedNotes(
            notes=notes,
            consensus_facts=consensus_facts,
            conflicting_facts=conflicting_facts,
            evidence_gaps=evidence_gaps,
            overall_confidence=min(1.0, confidence),
        )


# ============================================================================
# Chain-of-Note Pipeline
# ============================================================================


class ChainOfNote:
    """
    Chain-of-Note RAG Pipeline.

    Processes documents sequentially, generating notes and aggregating
    evidence before generating the final answer.
    """

    ANSWER_TEMPLATE = """Based on the aggregated notes, answer the query.

Query: {query}

Aggregated Evidence:
- Consensus Facts: {consensus}
- Conflicting Information: {conflicts}
- Evidence Gaps: {gaps}
- Confidence Level: {confidence:.0%}

Individual Notes Summary:
{notes_summary}

Generate a comprehensive answer based on the evidence.
If there are conflicts, acknowledge them.
If there are gaps, mention what is unknown.

Answer:"""

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol | None = None,
        note_generator: NoteGenerator | None = None,
        note_aggregator: NoteAggregator | None = None,
    ):
        """
        Initialize Chain-of-Note.

        Args:
            llm: LLM provider
            retriever: Optional retriever
            note_generator: Note generator (defaults to LLM-based)
            note_aggregator: Note aggregator (defaults to LLM-based)
        """
        self.llm = llm
        self.retriever = retriever
        self.note_generator = note_generator or LLMNoteGenerator(llm)
        self.note_aggregator = note_aggregator or LLMNoteAggregator(llm)

    async def answer(
        self,
        query: str,
        documents: list[RetrievedDocument] | None = None,
        top_k: int = 5,
    ) -> ChainOfNoteResult:
        """
        Answer query using Chain-of-Note.

        Args:
            query: User query
            documents: Pre-retrieved documents (optional)
            top_k: Number of documents to retrieve

        Returns:
            ChainOfNoteResult with notes and answer
        """
        reasoning_chain: list[str] = []

        # Step 1: Retrieve documents if not provided
        if documents is None:
            if self.retriever:
                documents = await self.retriever.retrieve(query, top_k=top_k)
                reasoning_chain.append(f"Retrieved {len(documents)} documents")
            else:
                documents = []

        if not documents:
            return self._empty_result(query)

        # Step 2: Generate notes for each document
        notes: list[Note] = []
        for i, doc in enumerate(documents):
            note = await self.note_generator.generate_note(query, doc)
            notes.append(note)
            reasoning_chain.append(
                f"Note {i + 1}: {note.note_type.value} "
                f"(evidence: {note.evidence_strength.value})"
            )

        # Step 3: Aggregate notes
        aggregated = await self.note_aggregator.aggregate(query, notes)
        reasoning_chain.append(
            f"Aggregated: {len(aggregated.consensus_facts)} consensus facts, "
            f"{len(aggregated.conflicting_facts)} conflicts"
        )

        # Step 4: Generate final answer
        final_answer = await self._generate_answer(query, notes, aggregated)
        reasoning_chain.append("Generated final answer")

        return ChainOfNoteResult(
            query=query,
            notes=notes,
            aggregated=aggregated,
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            confidence=aggregated.overall_confidence,
        )

    async def _generate_answer(
        self,
        query: str,
        notes: list[Note],
        aggregated: AggregatedNotes,
    ) -> str:
        """Generate final answer from aggregated notes."""
        # Summarize individual notes
        notes_summary = "\n".join(
            f"- Doc {i + 1}: {note.summary[:100]}..."
            for i, note in enumerate(notes)
            if note.note_type != NoteType.IRRELEVANT
        )

        prompt = self.ANSWER_TEMPLATE.format(
            query=query,
            consensus=", ".join(aggregated.consensus_facts) or "None identified",
            conflicts=", ".join(aggregated.conflicting_facts) or "None found",
            gaps=", ".join(aggregated.evidence_gaps) or "None identified",
            confidence=aggregated.overall_confidence,
            notes_summary=notes_summary or "No relevant notes",
        )

        answer = await self.llm.generate(prompt, max_tokens=500, temperature=0.3)
        return answer.strip()

    def _empty_result(self, query: str) -> ChainOfNoteResult:
        """Return empty result when no documents available."""
        return ChainOfNoteResult(
            query=query,
            notes=[],
            aggregated=AggregatedNotes(
                notes=[],
                consensus_facts=[],
                conflicting_facts=[],
                evidence_gaps=["No documents retrieved"],
                overall_confidence=0.0,
            ),
            final_answer="Unable to answer: no relevant documents found.",
            reasoning_chain=["No documents available"],
            confidence=0.0,
        )


# ============================================================================
# Advanced Chain-of-Note Variants
# ============================================================================


class IterativeChainOfNote(ChainOfNote):
    """
    Iterative Chain-of-Note with multi-round refinement.

    Identifies evidence gaps and retrieves additional documents.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        max_iterations: int = 3,
        **kwargs: Any,
    ):
        """Initialize iterative Chain-of-Note."""
        super().__init__(llm, retriever, **kwargs)
        self.max_iterations = max_iterations

    async def answer(
        self,
        query: str,
        documents: list[RetrievedDocument] | None = None,
        top_k: int = 5,
    ) -> ChainOfNoteResult:
        """Answer with iterative refinement."""
        all_notes: list[Note] = []
        all_docs: list[RetrievedDocument] = []
        reasoning_chain: list[str] = []

        current_query = query

        for iteration in range(self.max_iterations):
            reasoning_chain.append(f"--- Iteration {iteration + 1} ---")

            # Retrieve documents
            if self.retriever:
                new_docs = await self.retriever.retrieve(current_query, top_k=top_k)
                # Filter already seen documents
                new_docs = [
                    d for d in new_docs if d.content not in {doc.content for doc in all_docs}
                ]
                all_docs.extend(new_docs)
                reasoning_chain.append(f"Retrieved {len(new_docs)} new documents")

            if not new_docs:
                reasoning_chain.append("No new documents found, stopping iteration")
                break

            # Generate notes
            for doc in new_docs:
                note = await self.note_generator.generate_note(query, doc)
                all_notes.append(note)

            # Aggregate all notes
            aggregated = await self.note_aggregator.aggregate(query, all_notes)

            # Check if we have enough evidence
            if aggregated.overall_confidence >= 0.8 and len(aggregated.evidence_gaps) == 0:
                reasoning_chain.append("Sufficient evidence gathered")
                break

            # Generate follow-up query for gaps
            if aggregated.evidence_gaps:
                current_query = await self._generate_followup_query(query, aggregated.evidence_gaps)
                reasoning_chain.append(f"Follow-up query: {current_query[:50]}...")

        # Final aggregation and answer
        final_aggregated = await self.note_aggregator.aggregate(query, all_notes)
        final_answer = await self._generate_answer(query, all_notes, final_aggregated)

        return ChainOfNoteResult(
            query=query,
            notes=all_notes,
            aggregated=final_aggregated,
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            confidence=final_aggregated.overall_confidence,
        )

    async def _generate_followup_query(
        self,
        original_query: str,
        gaps: list[str],
    ) -> str:
        """Generate follow-up query to address evidence gaps."""
        prompt = f"""Given the original query and identified evidence gaps,
generate a follow-up search query to find missing information.

Original Query: {original_query}

Evidence Gaps:
{chr(10).join(f'- {gap}' for gap in gaps)}

Follow-up Query:"""

        response = await self.llm.generate(prompt, max_tokens=100, temperature=0.5)
        return response.strip()


class ContrastiveChainOfNote(ChainOfNote):
    """
    Contrastive Chain-of-Note that explicitly compares documents.

    Generates comparative notes highlighting agreements and disagreements.
    """

    COMPARISON_TEMPLATE = """Compare the following two documents in relation to the query.

Query: {query}

Document 1:
{doc1}

Document 2:
{doc2}

Comparison:
1. Where do they agree?
2. Where do they disagree?
3. Which provides stronger evidence?

Comparison Notes:"""

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol | None = None,
        **kwargs: Any,
    ):
        """Initialize contrastive Chain-of-Note."""
        super().__init__(llm, retriever, **kwargs)

    async def answer(
        self,
        query: str,
        documents: list[RetrievedDocument] | None = None,
        top_k: int = 5,
    ) -> ChainOfNoteResult:
        """Answer with contrastive analysis."""
        # Get base result
        base_result = await super().answer(query, documents, top_k)

        if len(base_result.notes) < 2:
            return base_result

        # Generate pairwise comparisons for top relevant notes
        relevant_notes = [n for n in base_result.notes if n.note_type != NoteType.IRRELEVANT]

        if len(relevant_notes) < 2:
            return base_result

        comparisons = await self._generate_comparisons(query, relevant_notes[:4])

        # Update aggregated notes with comparisons
        base_result.aggregated.conflicting_facts.extend(comparisons.get("disagreements", []))
        base_result.aggregated.consensus_facts.extend(comparisons.get("agreements", []))

        # Re-generate answer with contrastive info
        base_result.final_answer = await self._generate_contrastive_answer(
            query, base_result.notes, base_result.aggregated, comparisons
        )

        return base_result

    async def _generate_comparisons(
        self,
        query: str,
        notes: list[Note],
    ) -> dict[str, list[str]]:
        """Generate pairwise comparisons."""
        agreements: list[str] = []
        disagreements: list[str] = []

        # Compare pairs
        for i in range(len(notes) - 1):
            for j in range(i + 1, len(notes)):
                comparison = await self._compare_pair(query, notes[i].document, notes[j].document)
                agreements.extend(comparison.get("agreements", []))
                disagreements.extend(comparison.get("disagreements", []))

        return {"agreements": agreements, "disagreements": disagreements}

    async def _compare_pair(
        self,
        query: str,
        doc1: RetrievedDocument,
        doc2: RetrievedDocument,
    ) -> dict[str, list[str]]:
        """Compare two documents."""
        prompt = self.COMPARISON_TEMPLATE.format(
            query=query,
            doc1=doc1.content[:1000],
            doc2=doc2.content[:1000],
        )

        response = await self.llm.generate(prompt, max_tokens=300, temperature=0.3)

        # Parse response
        agreements: list[str] = []
        disagreements: list[str] = []

        lines = response.split("\n")
        current_section = ""

        for line in lines:
            line = line.strip()
            if "agree" in line.lower():
                current_section = "agree"
            elif "disagree" in line.lower():
                current_section = "disagree"
            elif line.startswith("-") or line.startswith("•"):
                fact = line[1:].strip()
                if current_section == "agree":
                    agreements.append(fact)
                elif current_section == "disagree":
                    disagreements.append(fact)

        return {"agreements": agreements, "disagreements": disagreements}

    async def _generate_contrastive_answer(
        self,
        query: str,
        notes: list[Note],
        aggregated: AggregatedNotes,
        comparisons: dict[str, list[str]],
    ) -> str:
        """Generate answer with contrastive analysis."""
        prompt = f"""Answer the query based on the evidence, highlighting both
agreements and disagreements between sources.

Query: {query}

Points of Agreement:
{chr(10).join(f'- {a}' for a in comparisons.get('agreements', [])[:5]) or '- None identified'}

Points of Disagreement:
{chr(10).join(f'- {d}' for d in comparisons.get('disagreements', [])[:5]) or '- None found'}

Confidence Level: {aggregated.overall_confidence:.0%}

Balanced Answer (acknowledge both agreements and conflicts):"""

        answer = await self.llm.generate(prompt, max_tokens=500, temperature=0.3)
        return answer.strip()


# ============================================================================
# Note-Based Filtering
# ============================================================================


class NoteBasedFilter:
    """Filter documents based on note quality."""

    def __init__(
        self,
        min_relevance: float = 0.5,
        min_evidence_strength: EvidenceStrength = EvidenceStrength.WEAK,
    ):
        """Initialize filter criteria."""
        self.min_relevance = min_relevance
        self.min_evidence_strength = min_evidence_strength

    def filter_notes(self, notes: list[Note]) -> list[Note]:
        """Filter notes based on quality criteria."""
        strength_order = [
            EvidenceStrength.NONE,
            EvidenceStrength.WEAK,
            EvidenceStrength.MODERATE,
            EvidenceStrength.STRONG,
        ]

        min_strength_idx = strength_order.index(self.min_evidence_strength)

        return [
            note
            for note in notes
            if note.relevance_score >= self.min_relevance
            and strength_order.index(note.evidence_strength) >= min_strength_idx
        ]

    def rank_notes(self, notes: list[Note]) -> list[Note]:
        """Rank notes by quality."""
        strength_scores = {
            EvidenceStrength.STRONG: 1.0,
            EvidenceStrength.MODERATE: 0.7,
            EvidenceStrength.WEAK: 0.4,
            EvidenceStrength.NONE: 0.1,
        }

        def note_score(note: Note) -> float:
            return note.relevance_score * 0.5 + strength_scores.get(note.evidence_strength, 0) * 0.5

        return sorted(notes, key=note_score, reverse=True)


# ============================================================================
# Factory Functions
# ============================================================================


def create_chain_of_note(
    llm: LLMProtocol,
    retriever: RetrieverProtocol | None = None,
) -> ChainOfNote:
    """
    Create a Chain-of-Note pipeline.

    Args:
        llm: LLM provider
        retriever: Optional retriever

    Returns:
        Configured ChainOfNote
    """
    return ChainOfNote(llm, retriever)


def create_iterative_con(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    max_iterations: int = 3,
) -> IterativeChainOfNote:
    """
    Create an iterative Chain-of-Note pipeline.

    Args:
        llm: LLM provider
        retriever: Retriever (required for iteration)
        max_iterations: Maximum refinement iterations

    Returns:
        Configured IterativeChainOfNote
    """
    return IterativeChainOfNote(llm, retriever, max_iterations)


def create_contrastive_con(
    llm: LLMProtocol,
    retriever: RetrieverProtocol | None = None,
) -> ContrastiveChainOfNote:
    """
    Create a contrastive Chain-of-Note pipeline.

    Args:
        llm: LLM provider
        retriever: Optional retriever

    Returns:
        Configured ContrastiveChainOfNote
    """
    return ContrastiveChainOfNote(llm, retriever)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating Chain-of-Note."""
    # This would use actual implementations
    # llm = OpenAILLM(api_key="...")
    # retriever = VectorStoreRetriever(...)

    # Basic Chain-of-Note
    # con = create_chain_of_note(llm, retriever)
    # result = await con.answer("What are the effects of climate change?")

    # Iterative with gap-filling
    # iterative = create_iterative_con(llm, retriever, max_iterations=3)
    # result = await iterative.answer("Explain quantum computing applications")

    # Contrastive analysis
    # contrastive = create_contrastive_con(llm, retriever)
    # result = await contrastive.answer("Is coffee good for health?")

    print("Chain-of-Note Implementation Ready")
    print("=" * 50)
    print("Features:")
    print("- Structured note generation per document")
    print("- Relevance and evidence strength assessment")
    print("- Note aggregation with consensus/conflict detection")
    print("- Iterative refinement with evidence gap filling")
    print("- Contrastive analysis between documents")
    print("- Note-based filtering and ranking")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
