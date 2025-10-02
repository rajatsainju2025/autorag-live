# Evaluation

AutoRAG-Live provides comprehensive evaluation tools for measuring retrieval and generation performance.

## 📊 Evaluation Metrics

### Retrieval Metrics

#### Exact Match (EM)
Fraction of predictions that exactly match the ground truth.

```python
from autorag_live.evals.advanced_metrics import exact_match

prediction = "The sky is blue"
ground_truth = "The sky is blue"
score = exact_match(prediction, ground_truth)  # 1.0

prediction = "The sky appears blue"
ground_truth = "The sky is blue"
score = exact_match(prediction, ground_truth)  # 0.0
```

#### F1 Score
Harmonic mean of precision and recall based on token overlap.

```python
from autorag_live.evals.advanced_metrics import token_f1

prediction = "The quick brown fox"
ground_truth = "The quick brown dog"
score = token_f1(prediction, ground_truth)  # 0.75
```

#### BLEU Score
Bilingual Evaluation Understudy for text generation quality.

```python
from autorag_live.evals.advanced_metrics import bleu_score

prediction = "The cat is on the mat"
ground_truth = "The cat is on the mat"
score = bleu_score(prediction, ground_truth)  # 1.0
```

#### ROUGE Score
Recall-Oriented Understudy for Gisting Evaluation.

```python
from autorag_live.evals.advanced_metrics import rouge_score

prediction = "The quick brown fox jumps"
ground_truth = "The quick brown fox leaps"
score = rouge_score(prediction, ground_truth)  # ROUGE scores
```

### Semantic Metrics

#### Relevance
How relevant the prediction is to the query.

```python
from autorag_live.evals.llm_judge import LLMJudge

judge = LLMJudge()
relevance = judge.judge_answer_relevance(
    query="What is machine learning?",
    prediction="ML is a subset of AI that learns from data",
    context="Machine learning is a method of data analysis..."
)
```

#### Faithfulness
How faithful the prediction is to the provided context.

```python
faithfulness = judge.judge_faithfulness(
    prediction="ML learns from data",
    context="Machine learning algorithms learn patterns from data"
)
```

#### Consistency
How consistent the prediction is with the ground truth.

```python
consistency = judge.judge_consistency(
    prediction="AI is artificial intelligence",
    ground_truth="Artificial intelligence is AI"
)
```

## 🧪 Evaluation Suites

### Small Suite

Fast evaluation on a small curated dataset.

```python
from autorag_live.evals import small

# Run evaluation
results = small.run_small_suite(
    runs_dir="runs",
    judge_type="deterministic"
)

print(f"Results: {results['metrics']}")
# {'em': 0.85, 'f1': 0.82, 'relevance': 0.78, 'faithfulness': 0.88}
```

### Custom Evaluation

Create your own evaluation dataset.

```python
from autorag_live.evals.advanced_metrics import evaluate_qa_pairs

# Your QA pairs
qa_pairs = [
    {
        "question": "What is Python?",
        "answer": "Python is a programming language",
        "context": ["Python is a high-level programming language..."],
        "prediction": "Python is a programming language"
    }
]

results = evaluate_qa_pairs(qa_pairs)
print(f"Average F1: {results['avg_f1']}")
```

## 📈 Benchmarking

### Performance Benchmarks

Measure system performance under different loads.

```python
from autorag_live.evals import performance_benchmarks

# Benchmark retrieval speed
benchmark = performance_benchmarks.RetrievalBenchmark(
    corpus_sizes=[100, 1000, 10000],
    queries=["test query"] * 10
)

results = benchmark.run()
print(f"Latency: {results['latency_ms']}ms")
print(f"Throughput: {results['queries_per_sec']} qps")
```

### Memory Benchmarks

Monitor memory usage during evaluation.

```python
from autorag_live.evals.performance_benchmarks import MemoryBenchmark

benchmark = MemoryBenchmark()
benchmark.start_monitoring()

# Your code here
results = small.run_small_suite()

memory_stats = benchmark.stop_monitoring()
print(f"Peak memory: {memory_stats['peak_mb']} MB")
```

## 🔍 Disagreement Analysis

### Jaccard Similarity

Measure overlap between retriever results.

```python
from autorag_live.disagreement import metrics

bm25_results = ["doc1", "doc2", "doc3"]
dense_results = ["doc2", "doc3", "doc4"]

jaccard = metrics.jaccard_at_k(bm25_results, dense_results, k=3)
print(f"Jaccard similarity: {jaccard}")  # 0.5
```

### Kendall Tau Correlation

Measure ranking agreement between retrievers.

```python
kendall_tau = metrics.kendall_tau_at_k(
    bm25_results, dense_results,
    corpus=["doc1", "doc2", "doc3", "doc4", "doc5"]
)
print(f"Kendall tau: {kendall_tau}")
```

### Diversity Analysis

Analyze how different retrievers are.

```python
from autorag_live.disagreement import report

diversity_report = report.generate_diversity_report(
    retrievers=["bm25", "dense", "hybrid"],
    queries=["query1", "query2"],
    corpus=["doc1", "doc2", "doc3"]
)

print(diversity_report)
```

## 🤖 LLM Judges

### Deterministic Judge

Fast, rule-based evaluation.

```python
from autorag_live.evals.llm_judge import DeterministicJudge

judge = DeterministicJudge()
relevance = judge.judge_answer_relevance(query, prediction, context)
```

### LLM Judge

AI-powered evaluation using language models.

```python
from autorag_live.evals.llm_judge import LLMJudge

judge = LLMJudge(
    model="gpt-3.5-turbo",
    temperature=0.1,
    api_key="your-api-key"
)

relevance = judge.judge_answer_relevance(query, prediction, context)
faithfulness = judge.judge_faithfulness(prediction, context)
```

## 📊 Result Analysis

### Statistical Significance

Test if improvements are statistically significant.

```python
from autorag_live.evals.advanced_metrics import statistical_significance

baseline_scores = [0.75, 0.78, 0.76, 0.79, 0.77]
new_scores = [0.82, 0.85, 0.83, 0.86, 0.84]

significant, p_value = statistical_significance(baseline_scores, new_scores)
print(f"Statistically significant: {significant}, p-value: {p_value}")
```

### Trend Analysis

Analyze performance trends over time.

```python
from autorag_live.data import time_series

tracker = time_series.TimeSeriesTracker()

# Add evaluation results over time
for i in range(10):
    tracker.add_note("evaluation", {
        "f1": 0.75 + i * 0.01,
        "timestamp": f"2024-01-{i+1}"
    })

trends = tracker.analyze_trends(days=30)
print(f"Performance trend: {trends['slope']}")
```

## 📋 Evaluation Configuration

Configure evaluation in `config/evaluation.yaml`:

```yaml
evaluation:
  judge_type: "deterministic"    # or "llm"
  metrics:
    - "exact_match"
    - "f1"
    - "bleu"
    - "rouge"
    - "relevance"
    - "faithfulness"

  llm_judge:
    model: "gpt-3.5-turbo"
    temperature: 0.1
    max_tokens: 100

  small_suite:
    seed: 42
    num_samples: 100
    runs_dir: "runs"
```

## 🎯 Best Practices

### Evaluation Checklist

- [ ] Use multiple metrics (EM, F1, semantic)
- [ ] Evaluate on diverse queries
- [ ] Test statistical significance
- [ ] Monitor performance over time
- [ ] Include human evaluation
- [ ] Cross-validate results

### Common Pitfalls

```python
# ❌ Don't evaluate only on training data
results = evaluate_on_training_data()  # Biased!

# ✅ Use held-out test sets
results = evaluate_on_test_data()

# ❌ Don't rely on single metrics
score = exact_match_only()  # Incomplete

# ✅ Use comprehensive evaluation
scores = comprehensive_evaluation()  # EM, F1, semantic
```

### Production Monitoring

```python
class ProductionEvaluator:
    def __init__(self):
        self.metrics = []

    def evaluate_prediction(self, query, prediction, ground_truth, context):
        """Evaluate a single prediction."""
        em = exact_match(prediction, ground_truth)
        f1 = token_f1(prediction, ground_truth)

        self.metrics.append({
            "query": query,
            "em": em,
            "f1": f1,
            "timestamp": time.time()
        })

        # Alert on performance drops
        if self._detect_performance_drop():
            self._alert_team()

    def _detect_performance_drop(self):
        """Detect significant performance drops."""
        recent_metrics = self.metrics[-100:]  # Last 100 predictions
        avg_f1 = sum(m["f1"] for m in recent_metrics) / len(recent_metrics)

        # Compare to baseline
        return avg_f1 < self.baseline_f1 - self.threshold
```

## 📈 Reporting

### Generate Evaluation Reports

```python
from autorag_live.disagreement import report

# Generate comprehensive report
eval_report = report.generate_evaluation_report(
    results_dir="runs",
    output_format="html"
)

# Save report
with open("evaluation_report.html", "w") as f:
    f.write(eval_report)
```

### Visualize Results

```python
import matplotlib.pyplot as plt

def plot_evaluation_results(results):
    """Plot evaluation metrics."""
    metrics = ["em", "f1", "relevance", "faithfulness"]
    values = [results["metrics"][m] for m in metrics]

    plt.bar(metrics, values)
    plt.title("Evaluation Results")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()
```
