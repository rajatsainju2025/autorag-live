"""
Stepback Prompting Implementation.

Based on: "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"
(Zheng et al., 2023)

Stepback prompting improves reasoning by:
1. Abstracting the original question to higher-level principles/concepts
2. Reasoning about the abstraction first
3. Then applying the abstract reasoning to the specific question

This helps LLMs avoid getting stuck in specifics and access broader knowledge.
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


# ============================================================================
# Stepback Abstraction Types
# ============================================================================


class AbstractionType(Enum):
    """Types of abstraction for stepback."""

    CONCEPT = "concept"  # Abstract to underlying concepts
    PRINCIPLE = "principle"  # Abstract to principles/rules
    CATEGORY = "category"  # Abstract to category/class
    PROCESS = "process"  # Abstract to general process
    RELATION = "relation"  # Abstract to relationships
    DOMAIN = "domain"  # Abstract to domain knowledge
    CAUSAL = "causal"  # Abstract to cause-effect patterns


@dataclass
class StepbackConfig:
    """Configuration for stepback prompting."""

    abstraction_type: AbstractionType = AbstractionType.CONCEPT
    num_abstractions: int = 1
    use_retrieval: bool = True
    retrieve_for_abstract: bool = True
    retrieve_for_original: bool = True
    max_context_length: int = 4000
    combine_strategy: str = "sequential"  # sequential, parallel, hierarchical


@dataclass
class AbstractQuestion:
    """An abstracted question."""

    original: str
    abstracted: str
    abstraction_type: AbstractionType
    reasoning_hint: str = ""


@dataclass
class StepbackResult:
    """Result from stepback prompting."""

    original_question: str
    abstract_questions: list[AbstractQuestion]
    abstract_reasoning: str
    final_answer: str
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)


# ============================================================================
# Abstraction Generation
# ============================================================================


class AbstractionGenerator(ABC):
    """Abstract base for generating stepback abstractions."""

    @abstractmethod
    async def generate_abstraction(
        self,
        question: str,
        abstraction_type: AbstractionType,
    ) -> AbstractQuestion:
        """Generate an abstracted question."""
        ...


class LLMAbstractionGenerator(AbstractionGenerator):
    """LLM-based abstraction generator."""

    ABSTRACTION_PROMPTS = {
        AbstractionType.CONCEPT: """Given the following specific question, generate a more abstract question
that asks about the underlying concepts or principles needed to answer it.

Original Question: {question}

Think: What fundamental concepts does this question rely on?

Abstract Question (asking about the underlying concept):""",
        AbstractionType.PRINCIPLE: """Given the following specific question, step back and generate a question
about the general principles or rules that govern this scenario.

Original Question: {question}

Think: What principles or rules apply here?

Abstract Question (asking about the principle):""",
        AbstractionType.CATEGORY: """Given the following specific question, generate a broader question
about the category or type of thing being discussed.

Original Question: {question}

Think: What category or type does this belong to?

Abstract Question (asking about the category):""",
        AbstractionType.PROCESS: """Given the following specific question, generate a question about
the general process or methodology involved.

Original Question: {question}

Think: What general process is this part of?

Abstract Question (asking about the process):""",
        AbstractionType.RELATION: """Given the following specific question, generate a question about
the underlying relationships between entities or concepts.

Original Question: {question}

Think: What relationships are involved?

Abstract Question (asking about relationships):""",
        AbstractionType.DOMAIN: """Given the following specific question, generate a broader question
about the domain knowledge needed to understand this topic.

Original Question: {question}

Think: What domain knowledge is required?

Abstract Question (asking about domain knowledge):""",
        AbstractionType.CAUSAL: """Given the following specific question, generate a question about
the causal mechanisms or cause-effect relationships involved.

Original Question: {question}

Think: What causes and effects are at play?

Abstract Question (asking about causation):""",
    }

    REASONING_HINTS = {
        AbstractionType.CONCEPT: "Consider the fundamental concepts first, then apply to the specific case.",
        AbstractionType.PRINCIPLE: "Identify the governing principles, then reason about the specific scenario.",
        AbstractionType.CATEGORY: "Understand the broader category, then narrow to the specific instance.",
        AbstractionType.PROCESS: "Understand the general process, then locate the specific step.",
        AbstractionType.RELATION: "Map out the relationships, then trace the specific connection.",
        AbstractionType.DOMAIN: "Build domain understanding, then address the specific question.",
        AbstractionType.CAUSAL: "Understand the causal chain, then identify the specific link.",
    }

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def generate_abstraction(
        self,
        question: str,
        abstraction_type: AbstractionType,
    ) -> AbstractQuestion:
        """Generate an abstracted question using LLM."""
        prompt_template = self.ABSTRACTION_PROMPTS[abstraction_type]
        prompt = prompt_template.format(question=question)

        response = await self.llm.generate(prompt, max_tokens=200, temperature=0.3)

        # Clean up response
        abstract_question = response.strip()
        if abstract_question.startswith("Abstract Question"):
            abstract_question = abstract_question.split(":", 1)[-1].strip()

        return AbstractQuestion(
            original=question,
            abstracted=abstract_question,
            abstraction_type=abstraction_type,
            reasoning_hint=self.REASONING_HINTS[abstraction_type],
        )

    async def generate_multiple_abstractions(
        self,
        question: str,
        types: list[AbstractionType] | None = None,
    ) -> list[AbstractQuestion]:
        """Generate multiple abstractions of different types."""
        if types is None:
            types = [AbstractionType.CONCEPT, AbstractionType.PRINCIPLE]

        abstractions = []
        for abs_type in types:
            abstract = await self.generate_abstraction(question, abs_type)
            abstractions.append(abstract)

        return abstractions


# ============================================================================
# Abstract Reasoning
# ============================================================================


class AbstractReasoner:
    """Reasons about abstract questions before specific ones."""

    REASONING_TEMPLATE = """You are reasoning through a question using the stepback approach.

First, let's think about the abstract, high-level question:
{abstract_question}

{context}

Reasoning about the abstract question:"""

    APPLICATION_TEMPLATE = """Now, apply this abstract reasoning to answer the specific question.

Abstract Reasoning:
{abstract_reasoning}

Original Specific Question:
{original_question}

{specific_context}

Based on the abstract reasoning above, here is the answer to the specific question:"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def reason_abstract(
        self,
        abstract_question: AbstractQuestion,
        context: str = "",
    ) -> str:
        """Generate reasoning for abstract question."""
        context_section = f"\nRelevant Context:\n{context}" if context else ""

        prompt = self.REASONING_TEMPLATE.format(
            abstract_question=abstract_question.abstracted,
            context=context_section,
        )

        reasoning = await self.llm.generate(prompt, max_tokens=500, temperature=0.5)
        return reasoning.strip()

    async def apply_to_specific(
        self,
        abstract_reasoning: str,
        original_question: str,
        specific_context: str = "",
    ) -> str:
        """Apply abstract reasoning to specific question."""
        context_section = f"\nAdditional Context:\n{specific_context}" if specific_context else ""

        prompt = self.APPLICATION_TEMPLATE.format(
            abstract_reasoning=abstract_reasoning,
            original_question=original_question,
            specific_context=context_section,
        )

        answer = await self.llm.generate(prompt, max_tokens=500, temperature=0.3)
        return answer.strip()


# ============================================================================
# Stepback RAG Pipeline
# ============================================================================


class StepbackRAG:
    """
    Stepback Prompting RAG Pipeline.

    Combines stepback prompting with retrieval for improved reasoning.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol | None = None,
        config: StepbackConfig | None = None,
    ):
        """
        Initialize Stepback RAG.

        Args:
            llm: LLM provider
            retriever: Optional retriever for context
            config: Stepback configuration
        """
        self.llm = llm
        self.retriever = retriever
        self.config = config or StepbackConfig()
        self.abstractor = LLMAbstractionGenerator(llm)
        self.reasoner = AbstractReasoner(llm)

    async def answer(
        self,
        question: str,
        abstraction_types: list[AbstractionType] | None = None,
    ) -> StepbackResult:
        """
        Answer question using stepback prompting.

        Args:
            question: User question
            abstraction_types: Types of abstractions to generate

        Returns:
            StepbackResult with answer and reasoning
        """
        if abstraction_types is None:
            abstraction_types = [self.config.abstraction_type]

        reasoning_steps: list[str] = []
        all_docs: list[RetrievedDocument] = []

        # Step 1: Generate abstractions
        abstractions = await self.abstractor.generate_multiple_abstractions(
            question, abstraction_types[: self.config.num_abstractions]
        )

        reasoning_steps.append(
            f"Generated {len(abstractions)} abstraction(s): "
            + ", ".join(a.abstracted[:50] + "..." for a in abstractions)
        )

        # Step 2: Retrieve for abstract questions if enabled
        abstract_context = ""
        if self.config.use_retrieval and self.config.retrieve_for_abstract:
            abstract_docs = await self._retrieve_for_abstractions(abstractions)
            all_docs.extend(abstract_docs)
            abstract_context = self._format_context(abstract_docs)
            reasoning_steps.append(
                f"Retrieved {len(abstract_docs)} documents for abstract questions"
            )

        # Step 3: Reason about abstractions
        abstract_reasonings = []
        for abstract in abstractions:
            reasoning = await self.reasoner.reason_abstract(abstract, abstract_context)
            abstract_reasonings.append(reasoning)
            reasoning_steps.append(f"Reasoned about: {abstract.abstraction_type.value}")

        combined_reasoning = "\n\n".join(abstract_reasonings)

        # Step 4: Retrieve for original question if enabled
        specific_context = ""
        if self.config.use_retrieval and self.config.retrieve_for_original:
            original_docs = []
            if self.retriever:
                original_docs = await self.retriever.retrieve(question, top_k=3)
            all_docs.extend(original_docs)
            specific_context = self._format_context(original_docs)
            reasoning_steps.append(
                f"Retrieved {len(original_docs)} documents for original question"
            )

        # Step 5: Apply abstract reasoning to specific question
        final_answer = await self.reasoner.apply_to_specific(
            combined_reasoning, question, specific_context
        )
        reasoning_steps.append("Applied abstract reasoning to specific question")

        return StepbackResult(
            original_question=question,
            abstract_questions=abstractions,
            abstract_reasoning=combined_reasoning,
            final_answer=final_answer,
            retrieved_docs=all_docs,
            reasoning_steps=reasoning_steps,
        )

    async def _retrieve_for_abstractions(
        self,
        abstractions: list[AbstractQuestion],
    ) -> list[RetrievedDocument]:
        """Retrieve documents for abstract questions."""
        if not self.retriever:
            return []

        all_docs: list[RetrievedDocument] = []
        seen_content: set[str] = set()

        for abstract in abstractions:
            docs = await self.retriever.retrieve(abstract.abstracted, top_k=2)
            for doc in docs:
                if doc.content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.content)

        return all_docs

    def _format_context(
        self,
        documents: list[RetrievedDocument],
    ) -> str:
        """Format documents into context string."""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc.content[:500]}")

        return "\n\n".join(context_parts)


# ============================================================================
# Hierarchical Stepback (Multiple Levels)
# ============================================================================


@dataclass
class HierarchicalAbstraction:
    """Multi-level abstraction."""

    levels: list[AbstractQuestion]  # From specific to most abstract
    reasoning_chain: list[str]


class HierarchicalStepback:
    """
    Hierarchical Stepback with multiple abstraction levels.

    Creates a tower of abstractions from specific to very general.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol | None = None,
        max_levels: int = 3,
    ):
        """Initialize hierarchical stepback."""
        self.llm = llm
        self.retriever = retriever
        self.max_levels = max_levels
        self.abstractor = LLMAbstractionGenerator(llm)
        self.reasoner = AbstractReasoner(llm)

    async def answer(
        self,
        question: str,
    ) -> StepbackResult:
        """
        Answer using hierarchical stepback.

        Args:
            question: User question

        Returns:
            StepbackResult with multi-level reasoning
        """
        # Build abstraction hierarchy
        hierarchy = await self._build_hierarchy(question)

        # Reason from most abstract to specific
        reasoning_chain = await self._reason_down_hierarchy(hierarchy)

        # Final answer
        final_answer = await self._generate_final_answer(question, reasoning_chain)

        return StepbackResult(
            original_question=question,
            abstract_questions=hierarchy.levels,
            abstract_reasoning="\n\n".join(hierarchy.reasoning_chain),
            final_answer=final_answer,
            reasoning_steps=[
                f"Level {i + 1}: {q.abstracted[:50]}..." for i, q in enumerate(hierarchy.levels)
            ],
        )

    async def _build_hierarchy(
        self,
        question: str,
    ) -> HierarchicalAbstraction:
        """Build hierarchy of abstractions."""
        levels = []
        current_question = question

        abstraction_sequence = [
            AbstractionType.CONCEPT,
            AbstractionType.PRINCIPLE,
            AbstractionType.DOMAIN,
        ]

        for i in range(min(self.max_levels, len(abstraction_sequence))):
            abs_type = abstraction_sequence[i]
            abstract = await self.abstractor.generate_abstraction(current_question, abs_type)
            levels.append(abstract)
            current_question = abstract.abstracted

        return HierarchicalAbstraction(levels=levels, reasoning_chain=[])

    async def _reason_down_hierarchy(
        self,
        hierarchy: HierarchicalAbstraction,
    ) -> list[str]:
        """Reason from most abstract to most specific."""
        reasoning_chain = []
        context = ""

        # Start from most abstract
        for abstract in reversed(hierarchy.levels):
            # Optionally retrieve context
            if self.retriever:
                docs = await self.retriever.retrieve(abstract.abstracted, top_k=2)
                context = "\n".join(d.content[:300] for d in docs)

            reasoning = await self.reasoner.reason_abstract(abstract, context)
            reasoning_chain.append(reasoning)

        hierarchy.reasoning_chain = reasoning_chain
        return reasoning_chain

    async def _generate_final_answer(
        self,
        question: str,
        reasoning_chain: list[str],
    ) -> str:
        """Generate final answer from reasoning chain."""
        prompt = f"""Based on the following chain of reasoning from general to specific:

{chr(10).join(f'Level {i + 1}:{chr(10)}{r}' for i, r in enumerate(reasoning_chain))}

Now answer this specific question:
{question}

Final Answer:"""

        answer = await self.llm.generate(prompt, max_tokens=500, temperature=0.3)
        return answer.strip()


# ============================================================================
# Domain-Specific Stepback
# ============================================================================


class DomainStepback:
    """
    Domain-specific stepback prompting.

    Customizes abstraction based on domain knowledge.
    """

    DOMAIN_ABSTRACTIONS = {
        "science": {
            "pattern": r"(physics|chemistry|biology|experiment|theory)",
            "type": AbstractionType.PRINCIPLE,
            "template": "What scientific principles govern {topic}?",
        },
        "math": {
            "pattern": r"(calculate|equation|formula|prove|derive)",
            "type": AbstractionType.CONCEPT,
            "template": "What mathematical concepts underlie {topic}?",
        },
        "history": {
            "pattern": r"(when|historical|century|era|war)",
            "type": AbstractionType.CAUSAL,
            "template": "What historical causes and effects relate to {topic}?",
        },
        "programming": {
            "pattern": r"(code|function|algorithm|implement|debug)",
            "type": AbstractionType.PROCESS,
            "template": "What programming paradigms apply to {topic}?",
        },
        "business": {
            "pattern": r"(company|market|strategy|profit|customer)",
            "type": AbstractionType.RELATION,
            "template": "What business relationships affect {topic}?",
        },
    }

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol | None = None,
    ):
        """Initialize domain stepback."""
        self.llm = llm
        self.retriever = retriever
        self.base_stepback = StepbackRAG(llm, retriever)

    async def answer(
        self,
        question: str,
    ) -> StepbackResult:
        """
        Answer using domain-aware stepback.

        Args:
            question: User question

        Returns:
            StepbackResult with domain-specific reasoning
        """
        # Detect domain
        domain = self._detect_domain(question)

        # Get domain-specific abstraction type
        if domain and domain in self.DOMAIN_ABSTRACTIONS:
            abs_type = self.DOMAIN_ABSTRACTIONS[domain]["type"]
        else:
            abs_type = AbstractionType.CONCEPT

        return await self.base_stepback.answer(question, [abs_type])

    def _detect_domain(self, question: str) -> str | None:
        """Detect domain from question."""
        import re

        question_lower = question.lower()

        for domain, config in self.DOMAIN_ABSTRACTIONS.items():
            if re.search(config["pattern"], question_lower):
                return domain

        return None


# ============================================================================
# Stepback with Chain-of-Thought
# ============================================================================


class StepbackCoT:
    """
    Stepback prompting combined with Chain-of-Thought.

    Uses stepback to guide CoT reasoning.
    """

    COT_TEMPLATE = """Let's think step by step.

First, let me step back and consider the broader context:
Question: {question}
Abstract Question: {abstract_question}

Reasoning about the abstract question:
{abstract_reasoning}

Now, applying this to the specific question step by step:

Step 1: Identify what we know from the abstract reasoning
{step1}

Step 2: Apply to the specific case
{step2}

Step 3: Draw conclusions
{step3}

Final Answer: {answer}"""

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol | None = None,
    ):
        """Initialize Stepback-CoT."""
        self.llm = llm
        self.retriever = retriever
        self.abstractor = LLMAbstractionGenerator(llm)
        self.reasoner = AbstractReasoner(llm)

    async def answer(
        self,
        question: str,
    ) -> StepbackResult:
        """
        Answer using stepback-guided CoT.

        Args:
            question: User question

        Returns:
            StepbackResult with CoT reasoning
        """
        # Generate abstraction
        abstract = await self.abstractor.generate_abstraction(question, AbstractionType.CONCEPT)

        # Get context if retriever available
        context = ""
        docs: list[RetrievedDocument] = []
        if self.retriever:
            docs = await self.retriever.retrieve(question, top_k=3)
            context = "\n".join(d.content[:300] for d in docs)

        # Generate abstract reasoning
        abstract_reasoning = await self.reasoner.reason_abstract(abstract, context)

        # Generate CoT answer
        cot_prompt = f"""Using stepback prompting and chain-of-thought reasoning:

Original Question: {question}

Abstract Question: {abstract.abstracted}

Abstract Reasoning:
{abstract_reasoning}

Now, let's think step by step to answer the original question:

Step 1:"""

        cot_response = await self.llm.generate(cot_prompt, max_tokens=800, temperature=0.3)

        return StepbackResult(
            original_question=question,
            abstract_questions=[abstract],
            abstract_reasoning=abstract_reasoning,
            final_answer=cot_response.strip(),
            retrieved_docs=docs,
            reasoning_steps=[
                "Generated concept abstraction",
                "Reasoned about abstract question",
                "Applied CoT to specific question",
            ],
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_stepback_rag(
    llm: LLMProtocol,
    retriever: RetrieverProtocol | None = None,
    abstraction_type: AbstractionType = AbstractionType.CONCEPT,
) -> StepbackRAG:
    """
    Create a stepback RAG pipeline.

    Args:
        llm: LLM provider
        retriever: Optional retriever
        abstraction_type: Type of abstraction

    Returns:
        Configured StepbackRAG
    """
    config = StepbackConfig(abstraction_type=abstraction_type)
    return StepbackRAG(llm, retriever, config)


def create_hierarchical_stepback(
    llm: LLMProtocol,
    retriever: RetrieverProtocol | None = None,
    max_levels: int = 3,
) -> HierarchicalStepback:
    """
    Create a hierarchical stepback pipeline.

    Args:
        llm: LLM provider
        retriever: Optional retriever
        max_levels: Maximum abstraction levels

    Returns:
        Configured HierarchicalStepback
    """
    return HierarchicalStepback(llm, retriever, max_levels)


def create_domain_stepback(
    llm: LLMProtocol,
    retriever: RetrieverProtocol | None = None,
) -> DomainStepback:
    """
    Create a domain-aware stepback pipeline.

    Args:
        llm: LLM provider
        retriever: Optional retriever

    Returns:
        Configured DomainStepback
    """
    return DomainStepback(llm, retriever)


def create_stepback_cot(
    llm: LLMProtocol,
    retriever: RetrieverProtocol | None = None,
) -> StepbackCoT:
    """
    Create a stepback with chain-of-thought pipeline.

    Args:
        llm: LLM provider
        retriever: Optional retriever

    Returns:
        Configured StepbackCoT
    """
    return StepbackCoT(llm, retriever)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating stepback prompting."""
    # This would use actual implementations
    # llm = OpenAILLM(api_key="...")
    # retriever = VectorStoreRetriever(...)

    # Basic stepback
    # stepback = create_stepback_rag(llm, retriever)
    # result = await stepback.answer("What is the capital of France?")

    # Hierarchical stepback
    # hierarchical = create_hierarchical_stepback(llm, retriever)
    # result = await hierarchical.answer("Why did World War I start?")

    # Domain-aware stepback
    # domain = create_domain_stepback(llm, retriever)
    # result = await domain.answer("How do I implement a binary search tree?")

    # Stepback with CoT
    # cot = create_stepback_cot(llm, retriever)
    # result = await cot.answer("Explain quantum entanglement")

    print("Stepback Prompting Implementation Ready")
    print("=" * 50)
    print("Features:")
    print("- Multiple abstraction types (concept, principle, category, etc.)")
    print("- Hierarchical multi-level abstraction")
    print("- Domain-specific customization")
    print("- Chain-of-thought integration")
    print("- Retrieval-augmented stepback reasoning")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
