"""
Evaluation dataset generation utilities for AutoRAG-Live.

Provides tools for generating synthetic evaluation datasets
for RAG systems, including question-answer pairs and retrieval
test data.

Features:
- Question generation from documents
- Answer generation with source attribution
- Synthetic dataset creation
- Dataset augmentation
- Quality filtering

Example usage:
    >>> generator = DatasetGenerator()
    >>> dataset = generator.generate_from_documents(
    ...     documents=["Python was created in 1991...", ...],
    ...     questions_per_doc=5
    ... )
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Types of questions to generate."""
    
    FACTUAL = "factual"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    LIST = "list"
    YES_NO = "yes_no"
    MULTI_HOP = "multi_hop"


class DifficultyLevel(str, Enum):
    """Difficulty levels for questions."""
    
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class QAPair:
    """A question-answer pair."""
    
    question: str
    answer: str
    
    # Source information
    source_text: str = ""
    source_id: Optional[str] = None
    
    # Metadata
    question_type: QuestionType = QuestionType.FACTUAL
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    
    # Quality indicators
    quality_score: float = 1.0
    is_valid: bool = True
    
    # Additional info
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.source_id:
            self.source_id = hashlib.md5(self.source_text[:100].encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "source_text": self.source_text,
            "source_id": self.source_id,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value,
            "quality_score": self.quality_score,
            "is_valid": self.is_valid,
            "entities": self.entities,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAPair:
        """Create from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            source_text=data.get("source_text", ""),
            source_id=data.get("source_id"),
            question_type=QuestionType(data.get("question_type", "factual")),
            difficulty=DifficultyLevel(data.get("difficulty", "medium")),
            quality_score=data.get("quality_score", 1.0),
            is_valid=data.get("is_valid", True),
            entities=data.get("entities", []),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalSample:
    """A retrieval evaluation sample."""
    
    query: str
    relevant_docs: List[str]
    irrelevant_docs: List[str] = field(default_factory=list)
    
    # Relevance labels (if graded)
    relevance_labels: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    query_type: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """An evaluation dataset."""
    
    name: str
    qa_pairs: List[QAPair] = field(default_factory=list)
    retrieval_samples: List[RetrievalSample] = field(default_factory=list)
    
    # Dataset info
    version: str = "1.0"
    description: str = ""
    created_at: str = ""
    
    # Statistics
    @property
    def num_qa_pairs(self) -> int:
        return len(self.qa_pairs)
    
    @property
    def num_retrieval_samples(self) -> int:
        return len(self.retrieval_samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "retrieval_samples": [
                {
                    "query": s.query,
                    "relevant_docs": s.relevant_docs,
                    "irrelevant_docs": s.irrelevant_docs,
                    "relevance_labels": s.relevance_labels,
                }
                for s in self.retrieval_samples
            ],
        }
    
    def get_questions(self) -> List[str]:
        """Get all questions."""
        return [qa.question for qa in self.qa_pairs]
    
    def get_answers(self) -> List[str]:
        """Get all answers."""
        return [qa.answer for qa in self.qa_pairs]
    
    def filter_by_type(self, question_type: QuestionType) -> List[QAPair]:
        """Filter QA pairs by question type."""
        return [qa for qa in self.qa_pairs if qa.question_type == question_type]
    
    def filter_by_difficulty(self, difficulty: DifficultyLevel) -> List[QAPair]:
        """Filter QA pairs by difficulty."""
        return [qa for qa in self.qa_pairs if qa.difficulty == difficulty]


class BaseQuestionGenerator(ABC):
    """Base class for question generators."""
    
    @abstractmethod
    def generate(
        self,
        text: str,
        num_questions: int = 5,
    ) -> List[QAPair]:
        """Generate questions from text."""
        pass


class TemplateQuestionGenerator(BaseQuestionGenerator):
    """Generate questions using templates."""
    
    # Question templates by type
    TEMPLATES = {
        QuestionType.FACTUAL: [
            "What is {entity}?",
            "Who {verb} {entity}?",
            "When did {entity} {verb}?",
            "Where is {entity} located?",
            "How many {entity} are there?",
        ],
        QuestionType.DEFINITION: [
            "What is the definition of {entity}?",
            "How would you define {entity}?",
            "What does {entity} mean?",
        ],
        QuestionType.EXPLANATION: [
            "Why is {entity} important?",
            "How does {entity} work?",
            "Explain the concept of {entity}.",
        ],
        QuestionType.LIST: [
            "What are the types of {entity}?",
            "List the characteristics of {entity}.",
            "What features does {entity} have?",
        ],
        QuestionType.YES_NO: [
            "Is {entity} {adjective}?",
            "Does {entity} {verb}?",
            "Can {entity} be {verb}?",
        ],
    }
    
    def __init__(self):
        """Initialize template generator."""
        pass
    
    def generate(
        self,
        text: str,
        num_questions: int = 5,
    ) -> List[QAPair]:
        """Generate questions using templates."""
        # Extract entities and key info
        entities = self._extract_entities(text)
        verbs = self._extract_verbs(text)
        sentences = self._split_sentences(text)
        
        questions = []
        
        for _ in range(num_questions):
            if not entities:
                break
            
            # Pick random entity and question type
            entity = random.choice(entities)
            q_type = random.choice(list(QuestionType))
            
            if q_type in self.TEMPLATES:
                template = random.choice(self.TEMPLATES[q_type])
                
                # Fill template
                question = template.format(
                    entity=entity,
                    verb=random.choice(verbs) if verbs else "is",
                    adjective="significant",
                )
                
                # Find answer from sentences
                answer = self._find_answer(question, entity, sentences)
                
                if question and answer:
                    questions.append(QAPair(
                        question=question,
                        answer=answer,
                        source_text=text,
                        question_type=q_type,
                    ))
        
        return questions
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities."""
        # Simple: extract capitalized phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter common words
        common = {"The", "This", "That", "It", "They", "He", "She", "We", "I"}
        entities = [e for e in entities if e not in common]
        
        return list(set(entities))
    
    def _extract_verbs(self, text: str) -> List[str]:
        """Extract verbs (simple heuristic)."""
        common_verbs = [
            "is", "was", "are", "were", "has", "have", "had",
            "created", "developed", "uses", "provides", "includes",
        ]
        found = []
        lower = text.lower()
        for verb in common_verbs:
            if verb in lower:
                found.append(verb)
        return found or ["is"]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_answer(
        self,
        question: str,
        entity: str,
        sentences: List[str],
    ) -> str:
        """Find answer sentence for entity."""
        entity_lower = entity.lower()
        
        for sentence in sentences:
            if entity_lower in sentence.lower():
                return sentence
        
        return sentences[0] if sentences else ""


class LLMQuestionGenerator(BaseQuestionGenerator):
    """Generate questions using LLM."""
    
    def __init__(
        self,
        llm_func: Callable[[str], str],
    ):
        """
        Initialize LLM generator.
        
        Args:
            llm_func: Function to call LLM
        """
        self.llm_func = llm_func
    
    def generate(
        self,
        text: str,
        num_questions: int = 5,
    ) -> List[QAPair]:
        """Generate questions using LLM."""
        prompt = self._build_prompt(text, num_questions)
        
        try:
            response = self.llm_func(prompt)
            return self._parse_response(response, text)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to template
            return TemplateQuestionGenerator().generate(text, num_questions)
    
    def _build_prompt(self, text: str, num_questions: int) -> str:
        """Build generation prompt."""
        return f"""Generate {num_questions} question-answer pairs based on the following text.
For each pair, provide:
- A clear, specific question
- The correct answer based only on the text

Text:
{text}

Generate the Q&A pairs in this format:
Q1: [question]
A1: [answer]

Q2: [question]
A2: [answer]

etc.
"""
    
    def _parse_response(self, response: str, source_text: str) -> List[QAPair]:
        """Parse LLM response."""
        pairs = []
        
        # Parse Q/A pattern
        q_pattern = re.compile(r'Q\d+:\s*(.+?)(?=A\d+:|$)', re.DOTALL)
        a_pattern = re.compile(r'A\d+:\s*(.+?)(?=Q\d+:|$)', re.DOTALL)
        
        questions = q_pattern.findall(response)
        answers = a_pattern.findall(response)
        
        for q, a in zip(questions, answers):
            q = q.strip()
            a = a.strip()
            
            if q and a:
                pairs.append(QAPair(
                    question=q,
                    answer=a,
                    source_text=source_text,
                ))
        
        return pairs


class DatasetGenerator:
    """
    Main dataset generation interface.
    
    Example:
        >>> generator = DatasetGenerator()
        >>> 
        >>> # Generate from documents
        >>> dataset = generator.generate_from_documents(
        ...     documents=["Doc 1 content...", "Doc 2 content..."],
        ...     questions_per_doc=5
        ... )
        >>> 
        >>> # Export
        >>> data = dataset.to_dict()
    """
    
    def __init__(
        self,
        question_generator: Optional[BaseQuestionGenerator] = None,
        quality_threshold: float = 0.5,
    ):
        """
        Initialize dataset generator.
        
        Args:
            question_generator: Question generator to use
            quality_threshold: Minimum quality score to include
        """
        self.question_generator = question_generator or TemplateQuestionGenerator()
        self.quality_threshold = quality_threshold
        self._quality_checker = QualityChecker()
    
    def generate_from_documents(
        self,
        documents: List[str],
        questions_per_doc: int = 5,
        dataset_name: str = "generated_dataset",
    ) -> EvalDataset:
        """
        Generate dataset from documents.
        
        Args:
            documents: Source documents
            questions_per_doc: Questions per document
            dataset_name: Name for the dataset
            
        Returns:
            EvalDataset
        """
        all_pairs = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Generating questions for document {i+1}/{len(documents)}")
            
            # Generate Q&A pairs
            pairs = self.question_generator.generate(doc, questions_per_doc)
            
            # Quality check
            for pair in pairs:
                pair.quality_score = self._quality_checker.score(pair)
                pair.is_valid = pair.quality_score >= self.quality_threshold
                
                if pair.is_valid:
                    all_pairs.append(pair)
        
        dataset = EvalDataset(
            name=dataset_name,
            qa_pairs=all_pairs,
        )
        
        logger.info(f"Generated {len(all_pairs)} valid Q&A pairs")
        return dataset
    
    def generate_retrieval_samples(
        self,
        documents: List[str],
        num_samples: int = 100,
        negatives_per_sample: int = 5,
    ) -> List[RetrievalSample]:
        """
        Generate retrieval evaluation samples.
        
        Args:
            documents: Corpus documents
            num_samples: Number of samples to generate
            negatives_per_sample: Negative docs per sample
            
        Returns:
            List of RetrievalSample
        """
        samples = []
        
        for _ in range(min(num_samples, len(documents))):
            # Pick a random document as relevant
            relevant_idx = random.randint(0, len(documents) - 1)
            relevant_doc = documents[relevant_idx]
            
            # Generate query from document
            qa_pairs = self.question_generator.generate(relevant_doc, 1)
            if not qa_pairs:
                continue
            
            query = qa_pairs[0].question
            
            # Sample negative documents
            negative_indices = [
                i for i in range(len(documents))
                if i != relevant_idx
            ]
            negative_indices = random.sample(
                negative_indices,
                min(negatives_per_sample, len(negative_indices))
            )
            negative_docs = [documents[i] for i in negative_indices]
            
            sample = RetrievalSample(
                query=query,
                relevant_docs=[relevant_doc],
                irrelevant_docs=negative_docs,
            )
            samples.append(sample)
        
        return samples
    
    def augment_dataset(
        self,
        dataset: EvalDataset,
        augmentation_factor: int = 2,
    ) -> EvalDataset:
        """
        Augment dataset with variations.
        
        Args:
            dataset: Original dataset
            augmentation_factor: How many variations per sample
            
        Returns:
            Augmented dataset
        """
        augmenter = DataAugmenter()
        augmented_pairs = []
        
        for qa in dataset.qa_pairs:
            augmented_pairs.append(qa)
            
            # Generate variations
            variations = augmenter.augment(qa, augmentation_factor - 1)
            augmented_pairs.extend(variations)
        
        return EvalDataset(
            name=f"{dataset.name}_augmented",
            qa_pairs=augmented_pairs,
            version=dataset.version,
            description=f"{dataset.description} (augmented)",
        )


class QualityChecker:
    """Check quality of generated Q&A pairs."""
    
    def __init__(
        self,
        min_question_length: int = 10,
        max_question_length: int = 500,
        min_answer_length: int = 5,
        max_answer_length: int = 2000,
    ):
        """
        Initialize quality checker.
        
        Args:
            min_question_length: Minimum question length
            max_question_length: Maximum question length
            min_answer_length: Minimum answer length
            max_answer_length: Maximum answer length
        """
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
    
    def score(self, qa_pair: QAPair) -> float:
        """
        Score quality of Q&A pair.
        
        Args:
            qa_pair: Q&A pair to score
            
        Returns:
            Quality score (0-1)
        """
        scores = []
        
        # Length checks
        q_len_score = self._length_score(
            len(qa_pair.question),
            self.min_question_length,
            self.max_question_length
        )
        scores.append(q_len_score)
        
        a_len_score = self._length_score(
            len(qa_pair.answer),
            self.min_answer_length,
            self.max_answer_length
        )
        scores.append(a_len_score)
        
        # Question format check
        if qa_pair.question.strip().endswith("?"):
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Answer relevance (basic check)
        overlap = self._word_overlap(qa_pair.question, qa_pair.answer)
        scores.append(min(overlap * 2, 1.0))
        
        # Source grounding
        if qa_pair.source_text:
            source_overlap = self._text_in_source(qa_pair.answer, qa_pair.source_text)
            scores.append(source_overlap)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _length_score(self, length: int, min_len: int, max_len: int) -> float:
        """Score based on length."""
        if length < min_len:
            return length / min_len
        elif length > max_len:
            return max_len / length
        return 1.0
    
    def _word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        return overlap / len(words1)
    
    def _text_in_source(self, text: str, source: str) -> float:
        """Check if text is grounded in source."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        source_words = set(re.findall(r'\b\w+\b', source.lower()))
        
        if not words:
            return 0.0
        
        found = len(words & source_words)
        return found / len(words)


class DataAugmenter:
    """Augment Q&A pairs with variations."""
    
    def __init__(self):
        """Initialize augmenter."""
        self._paraphrase_patterns = [
            (r"What is (.+)\?", r"Can you explain \1?"),
            (r"What is (.+)\?", r"Define \1."),
            (r"Who (.+)\?", r"Which person \1?"),
            (r"When did (.+)\?", r"What time did \1?"),
            (r"How (.+)\?", r"In what way \1?"),
        ]
    
    def augment(
        self,
        qa_pair: QAPair,
        num_variations: int = 2,
    ) -> List[QAPair]:
        """
        Generate variations of Q&A pair.
        
        Args:
            qa_pair: Original pair
            num_variations: Number of variations
            
        Returns:
            List of variations
        """
        variations = []
        
        for _ in range(num_variations):
            # Try paraphrasing question
            new_question = self._paraphrase_question(qa_pair.question)
            
            if new_question != qa_pair.question:
                variations.append(QAPair(
                    question=new_question,
                    answer=qa_pair.answer,
                    source_text=qa_pair.source_text,
                    source_id=qa_pair.source_id,
                    question_type=qa_pair.question_type,
                    difficulty=qa_pair.difficulty,
                    metadata={"augmented_from": qa_pair.question},
                ))
        
        return variations
    
    def _paraphrase_question(self, question: str) -> str:
        """Simple rule-based paraphrasing."""
        for pattern, replacement in self._paraphrase_patterns:
            if re.match(pattern, question, re.IGNORECASE):
                return re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question


class HardNegativeMiner:
    """
    Mine hard negative examples for retrieval.
    
    Example:
        >>> miner = HardNegativeMiner()
        >>> hard_negatives = miner.mine(
        ...     query="machine learning",
        ...     positive_doc="ML is a subset of AI...",
        ...     candidate_docs=all_docs
        ... )
    """
    
    def __init__(
        self,
        similarity_func: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize hard negative miner.
        
        Args:
            similarity_func: Function to compute similarity
        """
        self.similarity_func = similarity_func or self._default_similarity
    
    def mine(
        self,
        query: str,
        positive_doc: str,
        candidate_docs: List[str],
        num_negatives: int = 5,
        min_similarity: float = 0.3,
        max_similarity: float = 0.8,
    ) -> List[str]:
        """
        Mine hard negative documents.
        
        Args:
            query: Query string
            positive_doc: Known relevant document
            candidate_docs: Pool of candidate documents
            num_negatives: Number of negatives to return
            min_similarity: Minimum similarity threshold
            max_similarity: Maximum similarity threshold
            
        Returns:
            List of hard negative documents
        """
        # Score all candidates
        scored = []
        for doc in candidate_docs:
            if doc == positive_doc:
                continue
            
            sim = self.similarity_func(query, doc)
            
            # Hard negatives: somewhat similar but not too similar
            if min_similarity <= sim <= max_similarity:
                scored.append((doc, sim))
        
        # Sort by similarity (descending) and take top
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored[:num_negatives]]
    
    def _default_similarity(self, text1: str, text2: str) -> float:
        """Default word overlap similarity."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# Convenience functions

def generate_qa_pairs(
    documents: List[str],
    questions_per_doc: int = 5,
) -> List[QAPair]:
    """
    Generate Q&A pairs from documents.
    
    Args:
        documents: Source documents
        questions_per_doc: Questions per document
        
    Returns:
        List of QAPair
    """
    generator = DatasetGenerator()
    dataset = generator.generate_from_documents(
        documents, questions_per_doc
    )
    return dataset.qa_pairs


def generate_retrieval_dataset(
    documents: List[str],
    num_samples: int = 100,
) -> List[RetrievalSample]:
    """
    Generate retrieval evaluation samples.
    
    Args:
        documents: Corpus documents
        num_samples: Number of samples
        
    Returns:
        List of RetrievalSample
    """
    generator = DatasetGenerator()
    return generator.generate_retrieval_samples(documents, num_samples)


def create_eval_dataset(
    name: str,
    documents: List[str],
    questions_per_doc: int = 5,
    include_retrieval: bool = True,
) -> EvalDataset:
    """
    Create a complete evaluation dataset.
    
    Args:
        name: Dataset name
        documents: Source documents
        questions_per_doc: Questions per document
        include_retrieval: Include retrieval samples
        
    Returns:
        EvalDataset
    """
    generator = DatasetGenerator()
    
    # Generate Q&A
    dataset = generator.generate_from_documents(
        documents, questions_per_doc, name
    )
    
    # Add retrieval samples
    if include_retrieval:
        dataset.retrieval_samples = generator.generate_retrieval_samples(
            documents, num_samples=len(documents)
        )
    
    return dataset
