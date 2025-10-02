import os
from collections import Counter
from typing import Dict, List

import yaml


def mine_synonyms_from_disagreements(
    bm25_results: List[str], dense_results: List[str], hybrid_results: List[str], min_freq: int = 2
) -> Dict[str, List[str]]:
    """
    Mines potential synonyms from disagreement patterns.
    Looks for n-grams that appear frequently in one retriever but not others.
    """
    all_docs = bm25_results + dense_results + hybrid_results

    # Extract n-grams (1-3 tokens)
    ngrams = []
    for doc in all_docs:
        tokens = doc.lower().split()
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 4, len(tokens) + 1)):
                ngrams.append(" ".join(tokens[i:j]))

    ngram_counts = Counter(ngrams)

    # Group by frequency patterns - this is a simplified heuristic
    synonym_groups: Dict[str, List[str]] = {}
    for ngram, count in ngram_counts.items():
        if count >= min_freq and len(ngram.split()) <= 2:
            # Simple grouping by first word as key
            key = ngram.split()[0]
            if key not in synonym_groups:
                synonym_groups[key] = []
            if ngram not in synonym_groups[key]:
                synonym_groups[key].append(ngram)

    # Filter groups with multiple terms
    return {k: v for k, v in synonym_groups.items() if len(v) > 1}


def load_terms_yaml(path: str = "terms.yaml") -> Dict[str, List[str]]:
    """Load synonym terms from YAML file."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_terms_yaml(terms: Dict[str, List[str]], path: str = "terms.yaml") -> None:
    """Save synonym terms to YAML file."""
    with open(path, "w") as f:
        yaml.safe_dump(terms, f, default_flow_style=False)


def update_terms_from_mining(
    mined_terms: Dict[str, List[str]], existing_path: str = "terms.yaml"
) -> Dict[str, List[str]]:
    """Update existing terms with newly mined ones."""
    existing = load_terms_yaml(existing_path)

    for key, values in mined_terms.items():
        if key in existing:
            # Merge and deduplicate
            existing[key] = list(set(existing[key] + values))
        else:
            existing[key] = values

    save_terms_yaml(existing, existing_path)
    return existing
