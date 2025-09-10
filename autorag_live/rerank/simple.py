from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict
import math
import random


Token = str


def _tokenize(text: str) -> List[Token]:
    return [t.lower() for t in text.split()]  # simple + deterministic


def _features(query: str, doc: str) -> Tuple[float, float, float]:
    """
    Very small, deterministic feature vector:
    - f1: token overlap count
    - f2: jaccard similarity of tokens
    - f3: cosine similarity of term frequency vectors
    """
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(doc)

    if not q_tokens or not d_tokens:
        return 0.0, 0.0, 0.0

    q_set = set(q_tokens)
    d_set = set(d_tokens)

    overlap = len(q_set & d_set)
    union = len(q_set | d_set)
    jacc = (overlap / union) if union else 0.0

    # cosine on tf vectors
    tf_q: Dict[str, int] = {}
    tf_d: Dict[str, int] = {}
    for t in q_tokens:
        tf_q[t] = tf_q.get(t, 0) + 1
    for t in d_tokens:
        tf_d[t] = tf_d.get(t, 0) + 1

    # dot
    dot = 0.0
    for t, c in tf_q.items():
        dot += c * tf_d.get(t, 0)
    # norms
    nq = math.sqrt(sum(c * c for c in tf_q.values()))
    nd = math.sqrt(sum(c * c for c in tf_d.values()))
    cos = (dot / (nq * nd)) if (nq and nd) else 0.0

    return float(overlap), float(jacc), float(cos)


@dataclass
class SimpleReranker:
    seed: int = 42
    lr: float = 0.05
    epochs: int = 5

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.w1 = 0.1  # weight for overlap
        self.w2 = 0.1  # weight for jaccard
        self.w3 = 0.1  # weight for cosine

    def score(self, query: str, doc: str) -> float:
        f1, f2, f3 = _features(query, doc)
        return self.w1 * f1 + self.w2 * f2 + self.w3 * f3

    def fit(self, pairs: Sequence[Dict[str, object]]) -> None:
        """
        Train on a list of items with structure:
        {"query": str, "positive": str, "negatives": List[str]}
        Objective: score(q, pos) > score(q, neg) by a margin (hinge-like).
        Deterministic small number of epochs/steps for reproducibility.
        """
        margin = 0.5
        for _ in range(self.epochs):
            for item in pairs:
                q = str(item["query"])  # type: ignore[index]
                pos = str(item["positive"])  # type: ignore[index]
                negs = list(item.get("negatives", []))  # type: ignore[assignment]
                if not negs:
                    continue
                # pick a deterministic negative: the first
                n = str(negs[0])

                sp = self.score(q, pos)
                sn = self.score(q, n)
                loss = max(0.0, margin - (sp - sn))
                if loss > 0:
                    # gradient wrt weights is proportional to feature diff
                    p_f1, p_f2, p_f3 = _features(q, pos)
                    n_f1, n_f2, n_f3 = _features(q, n)
                    g1 = (n_f1 - p_f1)
                    g2 = (n_f2 - p_f2)
                    g3 = (n_f3 - p_f3)
                    self.w1 -= self.lr * g1
                    self.w2 -= self.lr * g2
                    self.w3 -= self.lr * g3

    def rerank(self, query: str, docs: List[str], k: int | None = None) -> List[str]:
        ranked = sorted(docs, key=lambda d: self.score(query, d), reverse=True)
        return ranked if k is None else ranked[:k]
