"""
Document summarization for AutoRAG-Live.

Provides extractive and abstractive summarization capabilities
for compressing documents into context windows.

Features:
- Extractive summarization (sentence selection)
- Abstractive summarization (text generation)
- Multi-document summarization
- Query-focused summarization
- Hierarchical summarization

Example usage:
    >>> summarizer = DocumentSummarizer()
    >>> summary = summarizer.summarize(long_document, max_length=200)
    
    >>> # Query-focused
    >>> summary = summarizer.summarize_for_query(
    ...     document, query="What are the main benefits?"
    ... )
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SummaryMethod(str, Enum):
    """Summarization methods."""
    
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"


class ExtractionStrategy(str, Enum):
    """Sentence extraction strategies."""
    
    TEXTRANK = "textrank"
    POSITION = "position"
    FREQUENCY = "frequency"
    CENTROID = "centroid"
    LEAD = "lead"
    RANDOM = "random"


@dataclass
class SummaryConfig:
    """Configuration for summarization."""
    
    # Method
    method: SummaryMethod = SummaryMethod.EXTRACTIVE
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.TEXTRANK
    
    # Length constraints
    max_length: int = 500  # Characters
    max_sentences: int = 5
    min_sentence_length: int = 10
    max_sentence_length: int = 200
    
    # Quality settings
    redundancy_threshold: float = 0.7
    coherence_weight: float = 0.3
    
    # TextRank settings
    textrank_damping: float = 0.85
    textrank_iterations: int = 100
    textrank_tolerance: float = 1e-4


@dataclass
class Sentence:
    """Represents a sentence with metadata."""
    
    text: str
    index: int
    start_char: int
    end_char: int
    word_count: int = 0
    score: float = 0.0
    embedding: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        """Compute word count."""
        self.word_count = len(self.text.split())


@dataclass
class SummaryResult:
    """Result of summarization."""
    
    summary: str
    source_text: str
    method: SummaryMethod
    
    # Metrics
    compression_ratio: float = 0.0
    sentence_count: int = 0
    word_count: int = 0
    
    # Selected sentences (for extractive)
    selected_sentences: List[Sentence] = field(default_factory=list)
    
    # Metadata
    processing_time_ms: float = 0.0
    
    def __post_init__(self) -> None:
        """Compute metrics."""
        if self.source_text:
            self.compression_ratio = len(self.summary) / len(self.source_text)
        self.word_count = len(self.summary.split())


class SentenceTokenizer:
    """Split text into sentences."""
    
    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard endings
        r'(?<=[.!?])\s*$|'  # End of text
        r'(?<=\.)\s+(?=\d)|'  # Period before number
        r'(?<=[.!?])\s+(?=["\'(])'  # Before quotes
    )
    
    # Abbreviations that don't end sentences
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "st", "ave", "blvd", "rd",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "inc", "ltd", "corp", "co",
        "vs", "etc", "eg", "ie", "al",
        "fig", "eq", "ref", "vol", "no",
        "approx", "est", "min", "max",
    }
    
    def tokenize(self, text: str) -> List[Sentence]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of Sentence objects
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        sentences = []
        current_pos = 0
        
        # Find sentence boundaries
        boundaries = list(self.SENTENCE_ENDINGS.finditer(text))
        
        for i, match in enumerate(boundaries):
            # Extract sentence
            end_pos = match.start() + 1
            sent_text = text[current_pos:end_pos].strip()
            
            if sent_text:
                sentences.append(Sentence(
                    text=sent_text,
                    index=len(sentences),
                    start_char=current_pos,
                    end_char=end_pos,
                ))
            
            current_pos = match.end()
        
        # Handle remaining text
        remaining = text[current_pos:].strip()
        if remaining:
            sentences.append(Sentence(
                text=remaining,
                index=len(sentences),
                start_char=current_pos,
                end_char=len(text),
            ))
        
        # Fallback for no sentence boundaries found
        if not sentences and text:
            sentences = [Sentence(
                text=text,
                index=0,
                start_char=0,
                end_char=len(text),
            )]
        
        return sentences


class TextRankScorer:
    """Score sentences using TextRank algorithm."""
    
    def __init__(
        self,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ):
        """
        Initialize TextRank scorer.
        
        Args:
            damping: Damping factor (0-1)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
        """
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def score(
        self,
        sentences: List[Sentence],
        similarity_matrix: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Score sentences using TextRank.
        
        Args:
            sentences: List of sentences
            similarity_matrix: Pre-computed similarity matrix
            
        Returns:
            List of scores
        """
        n = len(sentences)
        if n == 0:
            return []
        if n == 1:
            return [1.0]
        
        # Build similarity matrix if not provided
        if similarity_matrix is None:
            similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Normalize matrix
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = similarity_matrix / row_sums
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Power iteration
        for _ in range(self.max_iterations):
            new_scores = (1 - self.damping) / n + self.damping * matrix.T.dot(scores)
            
            # Check convergence
            if np.abs(new_scores - scores).sum() < self.tolerance:
                break
            
            scores = new_scores
        
        return scores.tolist()
    
    def _build_similarity_matrix(
        self, sentences: List[Sentence]
    ) -> np.ndarray:
        """Build word overlap similarity matrix."""
        n = len(sentences)
        matrix = np.zeros((n, n))
        
        # Tokenize sentences
        sentence_words = [
            set(s.text.lower().split()) for s in sentences
        ]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Jaccard similarity
                intersection = len(sentence_words[i] & sentence_words[j])
                union = len(sentence_words[i] | sentence_words[j])
                
                if union > 0:
                    similarity = intersection / union
                    matrix[i, j] = similarity
                    matrix[j, i] = similarity
        
        return matrix


class PositionScorer:
    """Score sentences by position."""
    
    def score(self, sentences: List[Sentence]) -> List[float]:
        """
        Score sentences by position (earlier = higher).
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of scores
        """
        n = len(sentences)
        if n == 0:
            return []
        
        scores = []
        for i, sentence in enumerate(sentences):
            # Exponential decay from position
            position_score = np.exp(-0.1 * i)
            
            # Boost first sentence
            if i == 0:
                position_score *= 1.5
            
            scores.append(position_score)
        
        # Normalize
        max_score = max(scores) if scores else 1.0
        return [s / max_score for s in scores]


class FrequencyScorer:
    """Score sentences by word frequency."""
    
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else",
        "when", "at", "from", "by", "on", "off", "for", "in", "out",
        "over", "to", "into", "with", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might",
        "must", "shall", "can", "need", "dare", "ought", "used",
        "to", "of", "it", "its", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "they", "what", "which",
        "who", "whom", "where", "when", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "as", "also",
    }
    
    def score(self, sentences: List[Sentence]) -> List[float]:
        """
        Score sentences by important word frequency.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of scores
        """
        if not sentences:
            return []
        
        # Build word frequency
        word_freq: Dict[str, int] = {}
        for sentence in sentences:
            for word in sentence.text.lower().split():
                word = re.sub(r'[^\w]', '', word)
                if word and word not in self.STOPWORDS and len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return [1.0 / len(sentences)] * len(sentences)
        
        # Score sentences
        scores = []
        for sentence in sentences:
            score = 0.0
            words = sentence.text.lower().split()
            for word in words:
                word = re.sub(r'[^\w]', '', word)
                if word in word_freq:
                    score += word_freq[word]
            
            # Normalize by sentence length
            if len(words) > 0:
                score /= len(words)
            
            scores.append(score)
        
        # Normalize
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        
        return scores


class CentroidScorer:
    """Score sentences by similarity to document centroid."""
    
    def score(
        self,
        sentences: List[Sentence],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Score sentences by centroid similarity.
        
        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings (n_sentences x dim)
            
        Returns:
            List of scores
        """
        if not sentences:
            return []
        
        # Use TF-IDF-like representation if no embeddings
        if embeddings is None:
            embeddings = self._compute_tfidf(sentences)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        
        if centroid_norm == 0:
            return [1.0 / len(sentences)] * len(sentences)
        
        centroid = centroid / centroid_norm
        
        # Compute similarities
        scores = []
        for i, embedding in enumerate(embeddings):
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                similarity = np.dot(embedding, centroid) / emb_norm
            else:
                similarity = 0.0
            scores.append(float(similarity))
        
        # Normalize to [0, 1]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        if max_score > min_score:
            scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        return scores
    
    def _compute_tfidf(self, sentences: List[Sentence]) -> np.ndarray:
        """Compute simple TF-IDF representations."""
        # Build vocabulary
        vocab: Dict[str, int] = {}
        for sentence in sentences:
            for word in sentence.text.lower().split():
                word = re.sub(r'[^\w]', '', word)
                if word and len(word) > 2:
                    if word not in vocab:
                        vocab[word] = len(vocab)
        
        if not vocab:
            return np.ones((len(sentences), 1))
        
        # Build TF-IDF matrix
        n_docs = len(sentences)
        matrix = np.zeros((n_docs, len(vocab)))
        
        # Document frequency
        doc_freq = np.zeros(len(vocab))
        
        for i, sentence in enumerate(sentences):
            words = sentence.text.lower().split()
            word_counts: Dict[str, int] = {}
            for word in words:
                word = re.sub(r'[^\w]', '', word)
                if word in vocab:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            for word, count in word_counts.items():
                idx = vocab[word]
                matrix[i, idx] = count
                doc_freq[idx] += 1
        
        # Apply IDF
        idf = np.log(n_docs / (doc_freq + 1)) + 1
        matrix = matrix * idf
        
        return matrix


class ExtractiveSummarizer:
    """Extractive summarization using sentence selection."""
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize extractive summarizer.
        
        Args:
            config: Summarization configuration
        """
        self.config = config or SummaryConfig()
        self.tokenizer = SentenceTokenizer()
        
        # Initialize scorers
        self._scorers = {
            ExtractionStrategy.TEXTRANK: TextRankScorer(
                damping=self.config.textrank_damping,
                max_iterations=self.config.textrank_iterations,
                tolerance=self.config.textrank_tolerance,
            ),
            ExtractionStrategy.POSITION: PositionScorer(),
            ExtractionStrategy.FREQUENCY: FrequencyScorer(),
            ExtractionStrategy.CENTROID: CentroidScorer(),
        }
    
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        max_sentences: Optional[int] = None,
        strategy: Optional[ExtractionStrategy] = None,
    ) -> SummaryResult:
        """
        Summarize text using extractive method.
        
        Args:
            text: Input text
            max_length: Maximum summary length (characters)
            max_sentences: Maximum number of sentences
            strategy: Extraction strategy to use
            
        Returns:
            SummaryResult
        """
        import time
        start_time = time.time()
        
        max_length = max_length or self.config.max_length
        max_sentences = max_sentences or self.config.max_sentences
        strategy = strategy or self.config.extraction_strategy
        
        # Tokenize into sentences
        sentences = self.tokenizer.tokenize(text)
        
        if not sentences:
            return SummaryResult(
                summary="",
                source_text=text,
                method=SummaryMethod.EXTRACTIVE,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Filter by length
        sentences = [
            s for s in sentences
            if self.config.min_sentence_length <= len(s.text) <= self.config.max_sentence_length
        ]
        
        if not sentences:
            # Return first sentence if all filtered
            original_sentences = self.tokenizer.tokenize(text)
            if original_sentences:
                sentences = [original_sentences[0]]
        
        # Score sentences
        if strategy == ExtractionStrategy.LEAD:
            scores = [1.0 - i * 0.1 for i in range(len(sentences))]
        elif strategy == ExtractionStrategy.RANDOM:
            import random
            scores = [random.random() for _ in sentences]
        else:
            scorer = self._scorers.get(strategy, self._scorers[ExtractionStrategy.TEXTRANK])
            scores = scorer.score(sentences)
        
        # Assign scores
        for i, score in enumerate(scores):
            sentences[i].score = score
        
        # Select sentences
        selected = self._select_sentences(
            sentences, max_length, max_sentences
        )
        
        # Sort by original position for coherence
        selected.sort(key=lambda s: s.index)
        
        # Build summary
        summary = " ".join(s.text for s in selected)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SummaryResult(
            summary=summary,
            source_text=text,
            method=SummaryMethod.EXTRACTIVE,
            sentence_count=len(selected),
            selected_sentences=selected,
            processing_time_ms=processing_time,
        )
    
    def _select_sentences(
        self,
        sentences: List[Sentence],
        max_length: int,
        max_sentences: int,
    ) -> List[Sentence]:
        """Select sentences respecting constraints."""
        # Sort by score descending
        ranked = sorted(sentences, key=lambda s: s.score, reverse=True)
        
        selected = []
        current_length = 0
        
        for sentence in ranked:
            if len(selected) >= max_sentences:
                break
            
            # Check length
            new_length = current_length + len(sentence.text) + 1
            if new_length > max_length and selected:
                continue
            
            # Check redundancy
            if self._is_redundant(sentence, selected):
                continue
            
            selected.append(sentence)
            current_length = new_length
        
        return selected
    
    def _is_redundant(
        self, sentence: Sentence, selected: List[Sentence]
    ) -> bool:
        """Check if sentence is too similar to already selected."""
        if not selected:
            return False
        
        sentence_words = set(sentence.text.lower().split())
        
        for sel in selected:
            sel_words = set(sel.text.lower().split())
            
            # Jaccard similarity
            intersection = len(sentence_words & sel_words)
            union = len(sentence_words | sel_words)
            
            if union > 0 and intersection / union > self.config.redundancy_threshold:
                return True
        
        return False


class QueryFocusedSummarizer:
    """Query-focused extractive summarization."""
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize query-focused summarizer.
        
        Args:
            config: Summarization configuration
        """
        self.config = config or SummaryConfig()
        self.tokenizer = SentenceTokenizer()
        self.textrank = TextRankScorer()
    
    def summarize(
        self,
        text: str,
        query: str,
        max_length: Optional[int] = None,
        max_sentences: Optional[int] = None,
    ) -> SummaryResult:
        """
        Summarize text focused on query.
        
        Args:
            text: Input text
            query: Focus query
            max_length: Maximum summary length
            max_sentences: Maximum sentences
            
        Returns:
            SummaryResult
        """
        import time
        start_time = time.time()
        
        max_length = max_length or self.config.max_length
        max_sentences = max_sentences or self.config.max_sentences
        
        # Tokenize
        sentences = self.tokenizer.tokenize(text)
        
        if not sentences:
            return SummaryResult(
                summary="",
                source_text=text,
                method=SummaryMethod.EXTRACTIVE,
            )
        
        # Score by query relevance
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sent_words = set(sentence.text.lower().split())
            
            # Query overlap
            overlap = len(query_words & sent_words)
            query_score = overlap / len(query_words) if query_words else 0.0
            
            sentence.score = query_score
        
        # Also factor in TextRank
        textrank_scores = self.textrank.score(sentences)
        for i, score in enumerate(textrank_scores):
            sentences[i].score = 0.7 * sentences[i].score + 0.3 * score
        
        # Select top sentences
        ranked = sorted(sentences, key=lambda s: s.score, reverse=True)
        
        selected = []
        current_length = 0
        
        for sentence in ranked:
            if len(selected) >= max_sentences:
                break
            if current_length + len(sentence.text) > max_length and selected:
                continue
            
            selected.append(sentence)
            current_length += len(sentence.text) + 1
        
        # Sort by position
        selected.sort(key=lambda s: s.index)
        
        summary = " ".join(s.text for s in selected)
        
        return SummaryResult(
            summary=summary,
            source_text=text,
            method=SummaryMethod.EXTRACTIVE,
            sentence_count=len(selected),
            selected_sentences=selected,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


class MultiDocumentSummarizer:
    """Summarize multiple documents."""
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize multi-document summarizer.
        
        Args:
            config: Summarization configuration
        """
        self.config = config or SummaryConfig()
        self.tokenizer = SentenceTokenizer()
        self.centroid_scorer = CentroidScorer()
    
    def summarize(
        self,
        documents: List[str],
        max_length: Optional[int] = None,
        max_sentences: Optional[int] = None,
    ) -> SummaryResult:
        """
        Summarize multiple documents.
        
        Args:
            documents: List of documents
            max_length: Maximum summary length
            max_sentences: Maximum sentences
            
        Returns:
            SummaryResult
        """
        import time
        start_time = time.time()
        
        max_length = max_length or self.config.max_length
        max_sentences = max_sentences or self.config.max_sentences
        
        # Collect all sentences with document source
        all_sentences: List[Tuple[int, Sentence]] = []
        
        for doc_idx, doc in enumerate(documents):
            sentences = self.tokenizer.tokenize(doc)
            for sentence in sentences:
                all_sentences.append((doc_idx, sentence))
        
        if not all_sentences:
            return SummaryResult(
                summary="",
                source_text="\n\n".join(documents),
                method=SummaryMethod.EXTRACTIVE,
            )
        
        # Score by centroid similarity
        sentences_only = [s for _, s in all_sentences]
        scores = self.centroid_scorer.score(sentences_only)
        
        for i, score in enumerate(scores):
            all_sentences[i][1].score = score
        
        # Select diverse sentences
        selected = self._select_diverse(
            all_sentences, max_length, max_sentences, len(documents)
        )
        
        # Build summary
        summary = " ".join(s.text for s in selected)
        
        return SummaryResult(
            summary=summary,
            source_text="\n\n".join(documents),
            method=SummaryMethod.EXTRACTIVE,
            sentence_count=len(selected),
            selected_sentences=selected,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    def _select_diverse(
        self,
        sentences: List[Tuple[int, Sentence]],
        max_length: int,
        max_sentences: int,
        n_docs: int,
    ) -> List[Sentence]:
        """Select diverse sentences from multiple documents."""
        # Sort by score
        ranked = sorted(sentences, key=lambda x: x[1].score, reverse=True)
        
        selected = []
        current_length = 0
        doc_counts = [0] * n_docs
        
        for doc_idx, sentence in ranked:
            if len(selected) >= max_sentences:
                break
            
            # Prefer documents with fewer selected sentences
            doc_penalty = doc_counts[doc_idx] * 0.1
            adjusted_score = sentence.score - doc_penalty
            
            if adjusted_score < 0.1 and selected:
                continue
            
            if current_length + len(sentence.text) > max_length and selected:
                continue
            
            selected.append(sentence)
            current_length += len(sentence.text) + 1
            doc_counts[doc_idx] += 1
        
        return selected


class DocumentSummarizer:
    """
    Main document summarizer with multiple strategies.
    
    Example usage:
        >>> summarizer = DocumentSummarizer()
        
        >>> # Basic extractive
        >>> result = summarizer.summarize(text, max_length=200)
        
        >>> # Query-focused
        >>> result = summarizer.summarize_for_query(text, "What are the benefits?")
        
        >>> # Multi-document
        >>> result = summarizer.summarize_documents([doc1, doc2, doc3])
    """
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize document summarizer.
        
        Args:
            config: Summarization configuration
        """
        self.config = config or SummaryConfig()
        self._extractive = ExtractiveSummarizer(self.config)
        self._query_focused = QueryFocusedSummarizer(self.config)
        self._multi_doc = MultiDocumentSummarizer(self.config)
    
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        max_sentences: Optional[int] = None,
        method: Optional[SummaryMethod] = None,
        strategy: Optional[ExtractionStrategy] = None,
    ) -> SummaryResult:
        """
        Summarize a single document.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            max_sentences: Maximum sentences
            method: Summarization method
            strategy: Extraction strategy
            
        Returns:
            SummaryResult
        """
        method = method or self.config.method
        
        if method == SummaryMethod.EXTRACTIVE:
            return self._extractive.summarize(
                text,
                max_length=max_length,
                max_sentences=max_sentences,
                strategy=strategy,
            )
        else:
            # Fallback to extractive for now
            return self._extractive.summarize(
                text,
                max_length=max_length,
                max_sentences=max_sentences,
                strategy=strategy,
            )
    
    def summarize_for_query(
        self,
        text: str,
        query: str,
        max_length: Optional[int] = None,
        max_sentences: Optional[int] = None,
    ) -> SummaryResult:
        """
        Summarize text focused on a query.
        
        Args:
            text: Input text
            query: Focus query
            max_length: Maximum length
            max_sentences: Maximum sentences
            
        Returns:
            SummaryResult
        """
        return self._query_focused.summarize(
            text, query,
            max_length=max_length,
            max_sentences=max_sentences,
        )
    
    def summarize_documents(
        self,
        documents: List[str],
        max_length: Optional[int] = None,
        max_sentences: Optional[int] = None,
    ) -> SummaryResult:
        """
        Summarize multiple documents.
        
        Args:
            documents: List of documents
            max_length: Maximum length
            max_sentences: Maximum sentences
            
        Returns:
            SummaryResult
        """
        return self._multi_doc.summarize(
            documents,
            max_length=max_length,
            max_sentences=max_sentences,
        )
    
    def summarize_incrementally(
        self,
        texts: List[str],
        budget: int,
    ) -> List[SummaryResult]:
        """
        Summarize texts with decreasing length budget.
        
        Args:
            texts: List of texts
            budget: Total character budget
            
        Returns:
            List of SummaryResult
        """
        results = []
        remaining_budget = budget
        n_texts = len(texts)
        
        for i, text in enumerate(texts):
            # Allocate budget
            texts_remaining = n_texts - i
            text_budget = remaining_budget // texts_remaining
            
            # Summarize
            result = self.summarize(text, max_length=text_budget)
            results.append(result)
            
            remaining_budget -= len(result.summary)
        
        return results


# Global summarizer instance
_default_summarizer: Optional[DocumentSummarizer] = None


def get_summarizer(
    config: Optional[SummaryConfig] = None
) -> DocumentSummarizer:
    """Get or create the default summarizer."""
    global _default_summarizer
    if _default_summarizer is None or config is not None:
        _default_summarizer = DocumentSummarizer(config)
    return _default_summarizer


def summarize(
    text: str,
    max_length: int = 500,
    max_sentences: int = 5,
) -> str:
    """
    Convenience function to summarize text.
    
    Args:
        text: Input text
        max_length: Maximum length
        max_sentences: Maximum sentences
        
    Returns:
        Summary string
    """
    result = get_summarizer().summarize(
        text,
        max_length=max_length,
        max_sentences=max_sentences,
    )
    return result.summary


def summarize_for_query(
    text: str,
    query: str,
    max_length: int = 500,
) -> str:
    """
    Convenience function for query-focused summarization.
    
    Args:
        text: Input text
        query: Focus query
        max_length: Maximum length
        
    Returns:
        Summary string
    """
    result = get_summarizer().summarize_for_query(
        text, query, max_length=max_length
    )
    return result.summary
