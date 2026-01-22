"""
Locality-Sensitive Hashing for fast document deduplication.

Uses MinHash LSH for near-duplicate detection with sub-linear time complexity.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


class MinHashLSH:
    """
    MinHash Locality-Sensitive Hashing for fast similarity detection.

    Enables O(1) average-case similarity search for near-duplicate detection,
    dramatically faster than pairwise comparison for large document sets.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        num_bands: int = 16,
    ):
        """
        Initialize MinHash LSH.

        Args:
            num_perm: Number of permutation functions (higher = more accurate)
            threshold: Similarity threshold for matches
            num_bands: Number of bands for LSH (higher = faster, less accurate)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        # LSH buckets: band_id -> {bucket_hash: [doc_ids]}
        self.buckets: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Document signatures: doc_id -> signature
        self.signatures: Dict[str, np.ndarray] = {}

    def _shingles(self, text: str, k: int = 3) -> Set[str]:
        """Generate character shingles from text."""
        text = text.lower().strip()
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def _minhash(self, shingles: Set[str]) -> np.ndarray:
        """Compute MinHash signature."""
        signature = np.full(self.num_perm, np.inf, dtype=np.float64)

        for shingle in shingles:
            # Use multiple hash functions
            for i in range(self.num_perm):
                h = hashlib.md5(f"{shingle}:{i}".encode()).hexdigest()
                hash_val = int(h, 16)
                signature[i] = min(signature[i], hash_val)

        return signature

    def _band_hash(self, signature: np.ndarray, band_idx: int) -> str:
        """Hash a band of the signature."""
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        band = signature[start:end]
        return hashlib.md5(band.tobytes()).hexdigest()

    def add(self, doc_id: str, text: str) -> None:
        """
        Add document to LSH index.

        Args:
            doc_id: Unique document identifier
            text: Document text
        """
        # Generate signature
        shingles = self._shingles(text)
        signature = self._minhash(shingles)
        self.signatures[doc_id] = signature

        # Add to LSH buckets
        for band_idx in range(self.num_bands):
            bucket_hash = self._band_hash(signature, band_idx)
            self.buckets[band_idx][bucket_hash].append(doc_id)

    def query(self, text: str) -> List[Tuple[str, float]]:
        """
        Find similar documents.

        Args:
            text: Query text

        Returns:
            List of (doc_id, similarity) pairs above threshold
        """
        # Generate query signature
        shingles = self._shingles(text)
        query_sig = self._minhash(shingles)

        # Find candidate documents from LSH buckets
        candidates: Set[str] = set()
        for band_idx in range(self.num_bands):
            bucket_hash = self._band_hash(query_sig, band_idx)
            if bucket_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][bucket_hash])

        # Compute exact similarities for candidates
        results = []
        for doc_id in candidates:
            doc_sig = self.signatures[doc_id]
            similarity = self._jaccard_similarity(query_sig, doc_sig)
            if similarity >= self.threshold:
                results.append((doc_id, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        return float(np.mean(sig1 == sig2))

    def deduplicate(self, documents: List[Tuple[str, str]]) -> List[str]:
        """
        Deduplicate a list of documents.

        Args:
            documents: List of (doc_id, text) tuples

        Returns:
            List of unique doc_ids (deduped)
        """
        seen: Set[str] = set()
        unique_ids: List[str] = []

        for doc_id, text in documents:
            # Check if similar document already seen
            similar = self.query(text)

            if not any(s_id in seen for s_id, _ in similar):
                # No duplicate found
                self.add(doc_id, text)
                seen.add(doc_id)
                unique_ids.append(doc_id)

        return unique_ids
