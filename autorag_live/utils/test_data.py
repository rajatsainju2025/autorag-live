"""Consolidated test and evaluation data to reduce duplication."""

from dataclasses import dataclass
from typing import List


@dataclass
class TestCorpus:
    """Standard test corpus used across evals and tests."""

    documents: List[str]
    name: str = "default"

    def __post_init__(self):
        """Validate corpus."""
        if not self.documents:
            raise ValueError("Corpus must have at least one document")


# Standard corpus used everywhere
STANDARD_CORPUS = TestCorpus(
    documents=[
        "The sky is blue.",
        "The sun is bright.",
        "The sun in the sky is bright.",
        "We can see the shining sun, the bright sun.",
        "The quick brown fox jumps over the lazy dog.",
        "A lazy fox is a sleeping fox.",
        "The fox is a mammal.",
        "Birds can fly in the sky.",
        "The blue sky stretches endlessly.",
        "Sunny weather brings joy.",
    ],
    name="standard",
)

# Extended corpus for larger tests
EXTENDED_CORPUS = TestCorpus(
    documents=STANDARD_CORPUS.documents
    + [
        "Programming is a valuable skill.",
        "Machine learning enables computers to learn.",
        "Natural language processing handles text.",
        "Deep learning uses neural networks.",
        "Artificial intelligence is transforming technology.",
    ],
    name="extended",
)


# Test queries with expected relevant documents
@dataclass
class TestQuery:
    """Query with expected relevant documents."""

    query: str
    relevant_indices: List[int]  # Indices into corpus
    description: str = ""


TEST_QUERIES = [
    TestQuery(
        query="What color is the sky?",
        relevant_indices=[0, 2, 8],
        description="Color-related query",
    ),
    TestQuery(
        query="Tell me about the sun",
        relevant_indices=[1, 2, 3, 9],
        description="Sun-related query",
    ),
    TestQuery(
        query="What animal is a fox?",
        relevant_indices=[4, 5, 6],
        description="Animal identification query",
    ),
    TestQuery(
        query="birds and flying",
        relevant_indices=[7],
        description="Bird/flying query",
    ),
]
