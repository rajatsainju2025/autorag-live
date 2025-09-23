"""
Configuration schemas for validation.
"""
from dataclasses import dataclass
from typing import List, Optional, Union
from omegaconf import MISSING


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class CacheConfig:
    enabled: bool = True
    max_size: int = 10000
    ttl: int = 3600


@dataclass
class BM25Config:
    b: float = 0.75
    k1: float = 1.5
    tokenizer: str = "whitespace"


@dataclass
class DenseConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class HybridConfig:
    weights: dict = MISSING
    normalize_scores: bool = True


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "autorag_docs"
    distance_metric: str = "cosine"


@dataclass
class ElasticsearchConfig:
    hosts: List[str] = MISSING
    index_name: str = "autorag_docs"
    timeout: int = 30


@dataclass
class RetrievalConfig:
    bm25: BM25Config = BM25Config()
    dense: DenseConfig = DenseConfig()
    hybrid: HybridConfig = HybridConfig()
    qdrant: QdrantConfig = QdrantConfig()
    elasticsearch: ElasticsearchConfig = ElasticsearchConfig()


@dataclass
class MetricConfig:
    name: str = MISSING
    enabled: bool = True
    judge_type: Optional[str] = None


@dataclass
class LLMJudgeConfig:
    type: str = "deterministic"
    temperature: float = 0.7
    max_tokens: int = 100


@dataclass
class BenchmarkConfig:
    warmup_iterations: int = 2
    test_iterations: int = 10
    memory_tracking: bool = True
    detailed_metrics: bool = True


@dataclass
class EvaluationConfig:
    metrics: List[MetricConfig] = MISSING
    llm_judge: LLMJudgeConfig = LLMJudgeConfig()
    benchmarks: BenchmarkConfig = BenchmarkConfig()


@dataclass
class AutoRAGConfig:
    name: str = "autorag-live"
    version: str = MISSING
    paths: dict = MISSING
    logging: LoggingConfig = LoggingConfig()
    cache: dict = MISSING
    retrieval: RetrievalConfig = RetrievalConfig()
    evaluation: EvaluationConfig = EvaluationConfig()