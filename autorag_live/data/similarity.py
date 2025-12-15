"""
Document similarity and deduplication for AutoRAG-Live.

Provides utilities for computing document similarity,
detecting duplicates, and clustering similar documents.

Features:
- Multiple similarity metrics (cosine, Jaccard, MinHash)
- Near-duplicate detection
- Document clustering
- Similarity-based filtering
- Efficient batch processing

Example usage:
    >>> detector = DuplicateDetector()
    >>> duplicates = detector.find_duplicates(documents)
    >>> unique_docs = detector.deduplicate(documents)
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SimilarityMetric(str, Enum):
    """Available similarity metrics."""
    
    COSINE = "cosine"
    JACCARD = "jaccard"
    MINHASH = "minhash"
    EDIT_DISTANCE = "edit_distance"
    OVERLAP = "overlap"
    DICE = "dice"


@dataclass
class SimilarityResult:
    """Result of a similarity computation."""
    
    doc1_id: str
    doc2_id: str
    score: float
    metric: SimilarityMetric
    
    # Additional info
    matched_features: int = 0
    total_features: int = 0
    
    @property
    def is_duplicate(self) -> bool:
        """Check if documents are likely duplicates."""
        return self.score >= 0.9
    
    @property
    def is_near_duplicate(self) -> bool:
        """Check if documents are near-duplicates."""
        return self.score >= 0.7


@dataclass
class Document:
    """Document representation for similarity."""
    
    id: str
    content: str
    
    # Computed features
    tokens: Optional[List[str]] = None
    shingles: Optional[Set[str]] = None
    minhash: Optional[List[int]] = None
    embedding: Optional[List[float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)


class TextPreprocessor:
    """Preprocess text for similarity computation."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        stem: bool = False,
    ):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove common stopwords
            stem: Apply stemming
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        
        self._stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might",
            "this", "that", "these", "those", "it", "its",
        }
    
    def preprocess(self, text: str) -> str:
        """Preprocess text."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        text = self.preprocess(text)
        tokens = text.split()
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        
        return tokens
    
    def get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """
        Get k-shingles (character n-grams).
        
        Args:
            text: Input text
            k: Shingle size
            
        Returns:
            Set of shingles
        """
        text = self.preprocess(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < k:
            return {text}
        
        return {text[i:i+k] for i in range(len(text) - k + 1)}
    
    def get_word_ngrams(self, text: str, n: int = 2) -> Set[str]:
        """
        Get word n-grams.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            Set of n-grams
        """
        tokens = self.tokenize(text)
        
        if len(tokens) < n:
            return {" ".join(tokens)}
        
        return {
            " ".join(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        }


class BaseSimilarityMetric(ABC):
    """Base class for similarity metrics."""
    
    @abstractmethod
    def compute(
        self,
        doc1: Document,
        doc2: Document,
    ) -> SimilarityResult:
        """Compute similarity between documents."""
        pass
    
    @abstractmethod
    def prepare(self, doc: Document) -> Document:
        """Prepare document for similarity computation."""
        pass


class JaccardSimilarity(BaseSimilarityMetric):
    """Jaccard similarity based on token/shingle overlap."""
    
    def __init__(
        self,
        use_shingles: bool = True,
        shingle_size: int = 3,
    ):
        """
        Initialize Jaccard similarity.
        
        Args:
            use_shingles: Use character shingles vs word tokens
            shingle_size: Size of shingles
        """
        self.use_shingles = use_shingles
        self.shingle_size = shingle_size
        self.preprocessor = TextPreprocessor()
    
    def prepare(self, doc: Document) -> Document:
        """Prepare document with shingles or tokens."""
        if self.use_shingles:
            doc.shingles = self.preprocessor.get_shingles(
                doc.content, self.shingle_size
            )
        else:
            doc.tokens = self.preprocessor.tokenize(doc.content)
        return doc
    
    def compute(
        self,
        doc1: Document,
        doc2: Document,
    ) -> SimilarityResult:
        """Compute Jaccard similarity."""
        if self.use_shingles:
            set1 = doc1.shingles or self.preprocessor.get_shingles(
                doc1.content, self.shingle_size
            )
            set2 = doc2.shingles or self.preprocessor.get_shingles(
                doc2.content, self.shingle_size
            )
        else:
            set1 = set(doc1.tokens or self.preprocessor.tokenize(doc1.content))
            set2 = set(doc2.tokens or self.preprocessor.tokenize(doc2.content))
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        score = intersection / union if union > 0 else 0.0
        
        return SimilarityResult(
            doc1_id=doc1.id,
            doc2_id=doc2.id,
            score=score,
            metric=SimilarityMetric.JACCARD,
            matched_features=intersection,
            total_features=union,
        )


class MinHashSimilarity(BaseSimilarityMetric):
    """MinHash-based similarity for efficient near-duplicate detection."""
    
    def __init__(
        self,
        num_hashes: int = 128,
        shingle_size: int = 3,
    ):
        """
        Initialize MinHash similarity.
        
        Args:
            num_hashes: Number of hash functions
            shingle_size: Shingle size for features
        """
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.preprocessor = TextPreprocessor()
        
        # Generate hash function parameters
        import random
        random.seed(42)
        self._hash_params = [
            (random.randint(1, 2**31), random.randint(0, 2**31))
            for _ in range(num_hashes)
        ]
    
    def _hash(self, value: str, a: int, b: int) -> int:
        """Apply a hash function."""
        h = int(hashlib.md5(value.encode()).hexdigest(), 16)
        return (a * h + b) % (2**31 - 1)
    
    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature."""
        signature = []
        
        for a, b in self._hash_params:
            min_hash = float('inf')
            for shingle in shingles:
                h = self._hash(shingle, a, b)
                min_hash = min(min_hash, h)
            signature.append(min_hash if min_hash != float('inf') else 0)
        
        return signature
    
    def prepare(self, doc: Document) -> Document:
        """Prepare document with MinHash signature."""
        shingles = self.preprocessor.get_shingles(doc.content, self.shingle_size)
        doc.shingles = shingles
        doc.minhash = self._compute_minhash(shingles)
        return doc
    
    def compute(
        self,
        doc1: Document,
        doc2: Document,
    ) -> SimilarityResult:
        """Compute MinHash similarity."""
        minhash1 = doc1.minhash
        minhash2 = doc2.minhash
        
        if not minhash1:
            doc1 = self.prepare(doc1)
            minhash1 = doc1.minhash
        
        if not minhash2:
            doc2 = self.prepare(doc2)
            minhash2 = doc2.minhash
        
        # Estimate Jaccard similarity from MinHash
        matches = sum(1 for h1, h2 in zip(minhash1, minhash2) if h1 == h2)
        score = matches / self.num_hashes
        
        return SimilarityResult(
            doc1_id=doc1.id,
            doc2_id=doc2.id,
            score=score,
            metric=SimilarityMetric.MINHASH,
            matched_features=matches,
            total_features=self.num_hashes,
        )


class CosineSimilarity(BaseSimilarityMetric):
    """Cosine similarity based on TF-IDF or embeddings."""
    
    def __init__(
        self,
        use_embeddings: bool = False,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize cosine similarity.
        
        Args:
            use_embeddings: Use embeddings vs TF-IDF
            embedding_func: Function to generate embeddings
        """
        self.use_embeddings = use_embeddings
        self.embedding_func = embedding_func
        self.preprocessor = TextPreprocessor()
        
        # For TF-IDF
        self._idf: Dict[str, float] = {}
        self._vocab: Set[str] = set()
    
    def prepare(self, doc: Document) -> Document:
        """Prepare document."""
        if self.use_embeddings and self.embedding_func:
            doc.embedding = self.embedding_func(doc.content)
        else:
            doc.tokens = self.preprocessor.tokenize(doc.content)
        return doc
    
    def compute(
        self,
        doc1: Document,
        doc2: Document,
    ) -> SimilarityResult:
        """Compute cosine similarity."""
        if self.use_embeddings:
            emb1 = doc1.embedding
            emb2 = doc2.embedding
            
            if not emb1 and self.embedding_func:
                emb1 = self.embedding_func(doc1.content)
            if not emb2 and self.embedding_func:
                emb2 = self.embedding_func(doc2.content)
            
            if emb1 and emb2:
                score = self._cosine(emb1, emb2)
            else:
                score = 0.0
        else:
            # TF-IDF based
            tokens1 = doc1.tokens or self.preprocessor.tokenize(doc1.content)
            tokens2 = doc2.tokens or self.preprocessor.tokenize(doc2.content)
            
            # Build term frequency vectors
            tf1 = self._term_freq(tokens1)
            tf2 = self._term_freq(tokens2)
            
            score = self._cosine_from_tf(tf1, tf2)
        
        return SimilarityResult(
            doc1_id=doc1.id,
            doc2_id=doc2.id,
            score=score,
            metric=SimilarityMetric.COSINE,
        )
    
    def _cosine(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _term_freq(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf: Dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        # Normalize
        total = sum(tf.values())
        if total > 0:
            tf = {k: v / total for k, v in tf.items()}
        
        return tf
    
    def _cosine_from_tf(
        self,
        tf1: Dict[str, float],
        tf2: Dict[str, float],
    ) -> float:
        """Compute cosine similarity from term frequencies."""
        all_terms = set(tf1.keys()) | set(tf2.keys())
        
        dot = sum(tf1.get(t, 0) * tf2.get(t, 0) for t in all_terms)
        norm1 = sum(v * v for v in tf1.values()) ** 0.5
        norm2 = sum(v * v for v in tf2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)


class DuplicateDetector:
    """
    Detect duplicate and near-duplicate documents.
    
    Example:
        >>> detector = DuplicateDetector(threshold=0.8)
        >>> 
        >>> # Find duplicates
        >>> duplicates = detector.find_duplicates(documents)
        >>> for pair in duplicates:
        ...     print(f"{pair.doc1_id} ~ {pair.doc2_id}: {pair.score:.2f}")
        >>> 
        >>> # Deduplicate
        >>> unique = detector.deduplicate(documents)
    """
    
    def __init__(
        self,
        metric: SimilarityMetric = SimilarityMetric.MINHASH,
        threshold: float = 0.8,
        num_hashes: int = 128,
        shingle_size: int = 3,
    ):
        """
        Initialize duplicate detector.
        
        Args:
            metric: Similarity metric to use
            threshold: Similarity threshold for duplicates
            num_hashes: Number of hashes for MinHash
            shingle_size: Shingle size
        """
        self.threshold = threshold
        
        if metric == SimilarityMetric.MINHASH:
            self._metric = MinHashSimilarity(num_hashes, shingle_size)
        elif metric == SimilarityMetric.JACCARD:
            self._metric = JaccardSimilarity(True, shingle_size)
        elif metric == SimilarityMetric.COSINE:
            self._metric = CosineSimilarity()
        else:
            self._metric = JaccardSimilarity(True, shingle_size)
    
    def find_duplicates(
        self,
        documents: List[Document],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SimilarityResult]:
        """
        Find all duplicate pairs.
        
        Args:
            documents: List of documents
            progress_callback: Optional progress callback
            
        Returns:
            List of duplicate pairs
        """
        # Prepare all documents
        prepared = [self._metric.prepare(doc) for doc in documents]
        
        duplicates = []
        total_pairs = len(prepared) * (len(prepared) - 1) // 2
        checked = 0
        
        for i in range(len(prepared)):
            for j in range(i + 1, len(prepared)):
                result = self._metric.compute(prepared[i], prepared[j])
                
                if result.score >= self.threshold:
                    duplicates.append(result)
                
                checked += 1
                if progress_callback and checked % 1000 == 0:
                    progress_callback(checked, total_pairs)
        
        return duplicates
    
    def deduplicate(
        self,
        documents: List[Document],
        keep: str = "first",
    ) -> List[Document]:
        """
        Remove duplicate documents.
        
        Args:
            documents: List of documents
            keep: Which duplicate to keep ("first", "last", "longest")
            
        Returns:
            Deduplicated document list
        """
        if not documents:
            return []
        
        # Find duplicates
        duplicates = self.find_duplicates(documents)
        
        # Build duplicate groups
        doc_map = {doc.id: doc for doc in documents}
        groups = self._build_groups(duplicates, set(doc_map.keys()))
        
        # Select representative from each group
        unique_ids = set()
        for group in groups:
            if len(group) == 1:
                unique_ids.add(list(group)[0])
            else:
                # Select based on keep strategy
                group_docs = [doc_map[doc_id] for doc_id in group]
                
                if keep == "longest":
                    selected = max(group_docs, key=lambda d: len(d.content))
                elif keep == "last":
                    selected = group_docs[-1]
                else:  # first
                    selected = group_docs[0]
                
                unique_ids.add(selected.id)
        
        return [doc for doc in documents if doc.id in unique_ids]
    
    def _build_groups(
        self,
        duplicates: List[SimilarityResult],
        all_ids: Set[str],
    ) -> List[Set[str]]:
        """Build groups of duplicate documents using Union-Find."""
        parent: Dict[str, str] = {doc_id: doc_id for doc_id in all_ids}
        
        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for dup in duplicates:
            union(dup.doc1_id, dup.doc2_id)
        
        # Build groups
        groups: Dict[str, Set[str]] = defaultdict(set)
        for doc_id in all_ids:
            groups[find(doc_id)].add(doc_id)
        
        return list(groups.values())


class LSHIndex:
    """
    Locality-Sensitive Hashing index for efficient similarity search.
    
    Example:
        >>> index = LSHIndex(num_bands=20)
        >>> index.build(documents)
        >>> 
        >>> # Find similar documents
        >>> similar = index.query(new_doc, k=5)
    """
    
    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        shingle_size: int = 3,
    ):
        """
        Initialize LSH index.
        
        Args:
            num_hashes: Number of hash functions
            num_bands: Number of bands for banding
            shingle_size: Shingle size
        """
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_size = shingle_size
        
        self._minhash = MinHashSimilarity(num_hashes, shingle_size)
        self._buckets: List[Dict[int, Set[str]]] = [
            {} for _ in range(num_bands)
        ]
        self._documents: Dict[str, Document] = {}
    
    def build(self, documents: List[Document]) -> None:
        """
        Build the LSH index.
        
        Args:
            documents: Documents to index
        """
        self._buckets = [{} for _ in range(self.num_bands)]
        self._documents = {}
        
        for doc in documents:
            self.add(doc)
    
    def add(self, doc: Document) -> None:
        """Add a document to the index."""
        # Compute MinHash
        prepared = self._minhash.prepare(doc)
        minhash = prepared.minhash
        
        self._documents[doc.id] = prepared
        
        # Add to bands
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(minhash[start:end])
            band_hash = hash(band_signature)
            
            if band_hash not in self._buckets[band_idx]:
                self._buckets[band_idx][band_hash] = set()
            self._buckets[band_idx][band_hash].add(doc.id)
    
    def query(
        self,
        doc: Document,
        k: int = 10,
    ) -> List[SimilarityResult]:
        """
        Find similar documents.
        
        Args:
            doc: Query document
            k: Number of results
            
        Returns:
            Similar documents with scores
        """
        # Compute MinHash for query
        prepared = self._minhash.prepare(doc)
        minhash = prepared.minhash
        
        # Find candidates from buckets
        candidates: Set[str] = set()
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(minhash[start:end])
            band_hash = hash(band_signature)
            
            if band_hash in self._buckets[band_idx]:
                candidates.update(self._buckets[band_idx][band_hash])
        
        # Remove self if present
        candidates.discard(doc.id)
        
        # Compute actual similarities
        results = []
        for cand_id in candidates:
            cand_doc = self._documents[cand_id]
            result = self._minhash.compute(prepared, cand_doc)
            results.append(result)
        
        # Sort and return top k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
    
    def find_near_duplicates(
        self,
        threshold: float = 0.8,
    ) -> List[SimilarityResult]:
        """
        Find all near-duplicate pairs in the index.
        
        Args:
            threshold: Similarity threshold
            
        Returns:
            List of near-duplicate pairs
        """
        # Collect all candidate pairs from buckets
        candidate_pairs: Set[Tuple[str, str]] = set()
        
        for bucket in self._buckets:
            for doc_ids in bucket.values():
                if len(doc_ids) > 1:
                    doc_list = sorted(doc_ids)
                    for i in range(len(doc_list)):
                        for j in range(i + 1, len(doc_list)):
                            candidate_pairs.add((doc_list[i], doc_list[j]))
        
        # Verify candidates
        results = []
        for id1, id2 in candidate_pairs:
            doc1 = self._documents[id1]
            doc2 = self._documents[id2]
            result = self._minhash.compute(doc1, doc2)
            
            if result.score >= threshold:
                results.append(result)
        
        return results


class DocumentClusterer:
    """
    Cluster similar documents.
    
    Example:
        >>> clusterer = DocumentClusterer(threshold=0.6)
        >>> clusters = clusterer.cluster(documents)
        >>> for i, cluster in enumerate(clusters):
        ...     print(f"Cluster {i}: {len(cluster)} documents")
    """
    
    def __init__(
        self,
        threshold: float = 0.6,
        metric: SimilarityMetric = SimilarityMetric.MINHASH,
    ):
        """
        Initialize clusterer.
        
        Args:
            threshold: Similarity threshold for clustering
            metric: Similarity metric
        """
        self.threshold = threshold
        self.detector = DuplicateDetector(
            metric=metric,
            threshold=threshold,
        )
    
    def cluster(
        self,
        documents: List[Document],
    ) -> List[List[Document]]:
        """
        Cluster similar documents.
        
        Args:
            documents: Documents to cluster
            
        Returns:
            List of document clusters
        """
        if not documents:
            return []
        
        # Find all similar pairs
        similar_pairs = self.detector.find_duplicates(documents)
        
        # Build adjacency list
        doc_map = {doc.id: doc for doc in documents}
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        for pair in similar_pairs:
            adjacency[pair.doc1_id].add(pair.doc2_id)
            adjacency[pair.doc2_id].add(pair.doc1_id)
        
        # Find connected components
        visited: Set[str] = set()
        clusters: List[List[Document]] = []
        
        for doc in documents:
            if doc.id in visited:
                continue
            
            # BFS to find component
            cluster_ids: Set[str] = set()
            queue = [doc.id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                cluster_ids.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            clusters.append([doc_map[doc_id] for doc_id in cluster_ids])
        
        return clusters


class ContentHasher:
    """
    Generate content hashes for exact duplicate detection.
    
    Example:
        >>> hasher = ContentHasher()
        >>> hash1 = hasher.hash("Hello world")
        >>> hash2 = hasher.hash("hello world")  # Same after normalization
    """
    
    def __init__(
        self,
        normalize: bool = True,
        algorithm: str = "sha256",
    ):
        """
        Initialize hasher.
        
        Args:
            normalize: Normalize text before hashing
            algorithm: Hash algorithm
        """
        self.normalize = normalize
        self.algorithm = algorithm
        self.preprocessor = TextPreprocessor()
    
    def hash(self, text: str) -> str:
        """
        Generate hash for text.
        
        Args:
            text: Input text
            
        Returns:
            Hash string
        """
        if self.normalize:
            text = self.preprocessor.preprocess(text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        h = hashlib.new(self.algorithm)
        h.update(text.encode('utf-8'))
        return h.hexdigest()
    
    def find_exact_duplicates(
        self,
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """
        Find exact duplicates by content hash.
        
        Args:
            documents: Documents to check
            
        Returns:
            Dict mapping hash to duplicate documents
        """
        hash_groups: Dict[str, List[Document]] = defaultdict(list)
        
        for doc in documents:
            doc_hash = self.hash(doc.content)
            hash_groups[doc_hash].append(doc)
        
        # Return only groups with duplicates
        return {
            h: docs for h, docs in hash_groups.items()
            if len(docs) > 1
        }


# Convenience functions

def find_duplicates(
    documents: List[Document],
    threshold: float = 0.8,
    metric: SimilarityMetric = SimilarityMetric.MINHASH,
) -> List[SimilarityResult]:
    """
    Find duplicate documents.
    
    Args:
        documents: Documents to check
        threshold: Similarity threshold
        metric: Similarity metric
        
    Returns:
        List of duplicate pairs
    """
    detector = DuplicateDetector(metric=metric, threshold=threshold)
    return detector.find_duplicates(documents)


def deduplicate(
    documents: List[Document],
    threshold: float = 0.8,
    keep: str = "first",
) -> List[Document]:
    """
    Remove duplicate documents.
    
    Args:
        documents: Documents to deduplicate
        threshold: Similarity threshold
        keep: Which duplicate to keep
        
    Returns:
        Deduplicated list
    """
    detector = DuplicateDetector(threshold=threshold)
    return detector.deduplicate(documents, keep=keep)


def compute_similarity(
    text1: str,
    text2: str,
    metric: SimilarityMetric = SimilarityMetric.JACCARD,
) -> float:
    """
    Compute similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        metric: Similarity metric
        
    Returns:
        Similarity score
    """
    doc1 = Document(id="1", content=text1)
    doc2 = Document(id="2", content=text2)
    
    if metric == SimilarityMetric.JACCARD:
        m = JaccardSimilarity()
    elif metric == SimilarityMetric.MINHASH:
        m = MinHashSimilarity()
    elif metric == SimilarityMetric.COSINE:
        m = CosineSimilarity()
    else:
        m = JaccardSimilarity()
    
    result = m.compute(m.prepare(doc1), m.prepare(doc2))
    return result.score
