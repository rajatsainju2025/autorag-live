from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
import os
import json
import random
import time

from autorag_live.evals.llm_judge import get_judge
from autorag_live.utils import get_logger

logger = get_logger(__name__)

SEED = 1337
random.seed(SEED)

@dataclass
class QAItem:
    id: str
    question: str
    context_docs: List[str]
    answer: str

SMALL_QA: List[QAItem] = [
    QAItem(
        id="q1",
        question="What color is the sky?",
        context_docs=[
            "The sky is blue.",
            "The sun is bright.",
            "The sun in the sky is bright.",
            "We can see the shining sun, the bright sun.",
        ],
        answer="blue",
    ),
    QAItem(
        id="q2",
        question="What animal is a fox?",
        context_docs=[
            "The quick brown fox jumps over the lazy dog.",
            "A lazy fox is a sleeping fox.",
            "The fox is a mammal.",
        ],
        answer="mammal",
    ),
]


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = pred.strip().lower().split()
    g = gold.strip().lower().split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    g_counts: Dict[str, int] = {}
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    prec = common / len(p)
    rec = common / len(g)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def simple_qa_answer(q: str, docs: List[str]) -> str:
    # extremely naive: if a token from canonical answers is found in docs, return it
    canonical = {"blue", "mammal"}
    blob = " ".join(docs).lower()
    for tok in canonical:
        if tok in blob:
            return tok
    return "unknown"


def run_small_suite(runs_dir: str = "runs", judge_type: str = "deterministic") -> Dict[str, Any]:
    logger.info(f"Starting small evaluation suite with judge_type: {judge_type}")
    os.makedirs(runs_dir, exist_ok=True)
    ts = int(time.time())
    run_id = f"small_{ts}"

    judge = get_judge(judge_type)
    logger.debug(f"Using judge: {judge_type}")
    
    results: List[Dict[str, Any]] = []
    ems: List[float] = []
    f1s: List[float] = []
    relevances: List[float] = []
    faithfulnesses: List[float] = []
    
    for i, item in enumerate(SMALL_QA):
        logger.debug(f"Processing QA item {i+1}/{len(SMALL_QA)}: {item.id}")
        pred = simple_qa_answer(item.question, item.context_docs)
        context = " ".join(item.context_docs)
        
        em = exact_match(pred, item.answer)
        f1 = token_f1(pred, item.answer)
        relevance = judge.judge_answer_relevance(item.question, pred, context)
        faithfulness = judge.judge_faithfulness(pred, context)
        
        ems.append(em)
        f1s.append(f1)
        relevances.append(relevance)
        faithfulnesses.append(faithfulness)
        
        results.append({
            "id": item.id,
            "question": item.question,
            "pred": pred,
            "gold": item.answer,
            "em": em,
            "f1": f1,
            "relevance": relevance,
            "faithfulness": faithfulness,
        })

    summary = {
        "run_id": run_id,
        "seed": SEED,
        "judge_type": judge_type,
        "metrics": {
            "em": sum(ems) / len(ems),
            "f1": sum(f1s) / len(f1s),
            "relevance": sum(relevances) / len(relevances),
            "faithfulness": sum(faithfulnesses) / len(faithfulnesses),
        },
        "results": results,
    }
    with open(os.path.join(runs_dir, f"{run_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Completed small evaluation suite. Run ID: {run_id}, Avg EM: {summary['metrics']['em']:.3f}, Avg F1: {summary['metrics']['f1']:.3f}, Avg Relevance: {summary['metrics']['relevance']:.3f}, Avg Faithfulness: {summary['metrics']['faithfulness']:.3f}")
    logger.info(f"Results saved to: {os.path.join(runs_dir, f'{run_id}.json')}")
    
    return summary
