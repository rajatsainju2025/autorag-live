# Configuration System

AutoRAG-Live uses a sophisticated configuration system built on [OmegaConf](https://omegaconf.readthedocs.io/) with dataclass schemas for type safety and validation.

## Overview

The configuration system provides:
- **Type-safe configuration** through dataclass schemas
- **Environment variable overrides** with automatic type conversion
- **Configuration validation** with detailed error messages
- **Version migration** for smooth upgrades
- **Hierarchical configuration** with component-specific configs

## 📁 Configuration Structure

Configurations are organized by component:

```
config/
├── retrieval.yaml      # Retriever configurations
├── evaluation.yaml     # Evaluation settings
├── pipeline.yaml       # Optimization pipeline settings
└── augmentation.yaml   # Data augmentation parameters
```

## 🔧 Retriever Configuration

### BM25 Configuration

```yaml
# config/retrieval.yaml
bm25:
  k1: 1.5                    # BM25 k1 parameter (1.2-2.0)
  b: 0.75                    # BM25 b parameter (0.0-1.0)
  epsilon: 0.25             # Minimum document frequency threshold
  tokenizer: "default"      # Tokenizer to use ("default", "whitespace", "nltk")
  lowercase: true           # Convert query to lowercase
  stop_words: "english"     # Stop words to remove ("english", "none", or list)
```

### Dense Retrieval Configuration

```yaml
# config/retrieval.yaml
dense:
  model_name: "all-MiniLM-L6-v2"  # Sentence transformer model
  device: "auto"                 # Device to run on ("auto", "cpu", "cuda")
  batch_size: 32                 # Batch size for encoding
  normalize_embeddings: true     # Normalize embeddings to unit length
  cache_dir: null                # Cache directory for models
  trust_remote_code: false       # Trust remote code in transformers
```

### Hybrid Retrieval Configuration

```yaml
# config/retrieval.yaml
hybrid:
  bm25_weight: 0.5              # Weight for BM25 scores (0.0-1.0)
  dense_weight: 0.5             # Weight for dense scores (0.0-1.0)
  normalize_weights: true        # Auto-normalize weights to sum to 1.0
  combination_method: "linear"   # How to combine scores ("linear", "harmonic")
```

### Vector Database Configurations

#### Qdrant

```yaml
# config/retrieval.yaml
qdrant:
  url: "http://localhost:6333"   # Qdrant server URL
  collection_name: "autorag"     # Collection name
  vector_size: 384              # Vector dimensionality
  distance: "Cosine"            # Distance metric ("Cosine", "Euclid", "Dot")
  on_disk: false                # Store vectors on disk
  hnsw_config:                  # HNSW index configuration
    m: 16                       # Number of connections per node
    ef_construct: 100           # Construction parameter
  optimizers_config:            # Optimization configuration
    deleted_threshold: 0.2      # Deletion threshold
    vacuum_min_vector_number: 1000
```

#### FAISS

```yaml
# config/retrieval.yaml
faiss:
  index_type: "IndexIVFFlat"     # FAISS index type
  nlist: 100                    # Number of clusters for IVF
  nprobe: 10                    # Number of clusters to search
  metric: "L2"                  # Distance metric ("L2", "IP")
  gpu: false                    # Use GPU acceleration
```

#### Elasticsearch

```yaml
# config/retrieval.yaml
elasticsearch:
  hosts: ["http://localhost:9200"]  # Elasticsearch hosts
  index_name: "autorag"            # Index name
  vector_field: "vector"           # Field name for vectors
  text_field: "text"               # Field name for text
  similarity: "cosine"             # Similarity metric
  settings:                        # Index settings
    number_of_shards: 1
    number_of_replicas: 0
  mappings:                        # Field mappings
    properties:
      vector:
        type: "dense_vector"
        dims: 384
      text:
        type: "text"
```

## 📊 Evaluation Configuration

```yaml
# config/evaluation.yaml
evaluation:
  judge_type: "deterministic"    # Judge type ("deterministic", "llm")
  metrics:                       # Metrics to compute
    - "exact_match"
    - "f1"
    - "relevance"
    - "faithfulness"
  llm_judge:                     # LLM judge configuration
    model: "gpt-3.5-turbo"       # LLM model for judging
    temperature: 0.1             # Temperature for generation
    max_tokens: 100              # Maximum tokens to generate
    api_key: null                # OpenAI API key (env: OPENAI_API_KEY)
  small_suite:                   # Small evaluation suite settings
    seed: 42                     # Random seed
    num_samples: 100             # Number of samples to evaluate
    runs_dir: "runs"             # Directory to save results
```

## 🔄 Pipeline Configuration

```yaml
# config/pipeline.yaml
pipeline:
  acceptance_policy:             # Acceptance policy settings
    threshold: 0.01              # Minimum improvement threshold
    metric_key: "f1"             # Metric to evaluate on
    best_runs_file: "best_runs.json"
  bandit_optimizer:              # Bandit optimization settings
    bandit_type: "ucb"           # Bandit algorithm ("ucb", "thompson")
    alpha: 1.0                   # Exploration parameter
    num_arms: 10                 # Number of arms to optimize
    max_iterations: 1000         # Maximum optimization iterations
  hybrid_optimizer:              # Hybrid weight optimization
    grid_size: 11                # Grid search resolution (odd numbers work best)
    diversity_weight: 0.7        # Weight for diversity vs performance
    performance_weight: 0.3      # Weight for performance vs diversity
```

## 🎯 Augmentation Configuration

```yaml
# config/augmentation.yaml
augmentation:
  hard_negatives:                # Hard negative mining
    num_negatives: 3             # Number of hard negatives per query
    similarity_threshold: 0.8    # Similarity threshold for hard negatives
    diversity_threshold: 0.3     # Diversity threshold for negatives
  query_rewrites:                # Query rewriting
    num_rewrites: 5              # Number of rewrites to generate
    rewrite_types:               # Types of rewrites to generate
      - "paraphrase"
      - "specificity"
      - "generalization"
    temperature: 0.7             # Temperature for generation
  synonym_miner:                 # Synonym mining
    min_frequency: 5             # Minimum frequency for synonyms
    max_synonyms: 10             # Maximum synonyms per term
    similarity_threshold: 0.85   # Similarity threshold for synonyms
```

## 🌍 Environment Variables

Override configuration with environment variables:

```bash
# Retriever settings
export AUTORAG_DENSE_MODEL="all-mpnet-base-v2"
export AUTORAG_DEVICE="cuda"

# Database connections
export AUTORAG_QDRANT_URL="http://localhost:6333"
export AUTORAG_ELASTICSEARCH_HOST="http://localhost:9200"

# API keys
export OPENAI_API_KEY="your-api-key-here"

# Logging
export AUTORAG_LOG_LEVEL="DEBUG"
export AUTORAG_LOG_FILE="logs/autorag.log"
```

## 🔧 Runtime Configuration

Modify configuration at runtime:

```python
from autorag_live.utils import ConfigManager

# Get configuration manager
config = ConfigManager()

# Load specific config
retrieval_config = config.get_config('retrieval')

# Modify settings
retrieval_config['dense']['batch_size'] = 64
retrieval_config['bm25']['k1'] = 1.8

# Save changes
config.save_config('retrieval', retrieval_config)
```

## 📝 Configuration Validation

AutoRAG-Live validates configuration on load:

```python
from autorag_live.utils import ConfigManager

try:
    config = ConfigManager()
    # Configuration loaded and validated
except ValueError as e:
    print(f"Configuration error: {e}")
```

## 🔄 Configuration Reload

Reload configuration without restarting:

```python
from autorag_live.utils import ConfigManager

config = ConfigManager()
# ... modify config files ...
config.reload()  # Reload all configurations
```

## 📋 Configuration Schema

View the complete configuration schema:

```python
from autorag_live.utils import schema
print(schema.CONFIG_SCHEMA)
```

## 🎯 Best Practices

### Performance Tuning

```yaml
# High performance configuration
dense:
  model_name: "all-mpnet-base-v2"  # Better model
  device: "cuda"                  # GPU acceleration
  batch_size: 64                  # Larger batches

bm25:
  k1: 1.2                        # Tuned BM25 parameters
  b: 0.8
```

### Memory Optimization

```yaml
# Memory-efficient configuration
dense:
  model_name: "all-MiniLM-L6-v2"  # Smaller model
  device: "cpu"                  # CPU only
  batch_size: 8                  # Smaller batches

faiss:
  index_type: "IndexIVFPQ"       # Compressed index
  nlist: 256                     # More clusters
```

### Development Configuration

```yaml
# Development-friendly configuration
evaluation:
  judge_type: "deterministic"    # Fast evaluation
  small_suite:
    num_samples: 10              # Quick evaluation

logging:
  level: "DEBUG"                 # Verbose logging
  console: true                  # Console output
```
