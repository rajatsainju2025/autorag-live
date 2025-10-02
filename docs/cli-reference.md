# CLI Reference

AutoRAG-Live provides a comprehensive command-line interface for all operations.

## 🚀 Getting Started

```bash
# Show help
autorag --help

# Show version
autorag --version
```

## 📊 Evaluation Commands

### Run Small Evaluation Suite

```bash
# Basic evaluation
autorag eval small

# With custom settings
autorag eval small --judge-type llm --runs-dir my_runs

# Specify number of samples
autorag eval small --num-samples 50
```

**Options:**
- `--judge-type`: Judge type (`deterministic`, `llm`) [default: deterministic]
- `--runs-dir`: Directory to save results [default: runs]
- `--num-samples`: Number of samples to evaluate [default: 100]
- `--seed`: Random seed [default: 42]

### Custom Evaluation

```bash
# Evaluate with custom dataset
autorag eval custom path/to/dataset.json

# With configuration file
autorag eval custom dataset.json --config config/evaluation.yaml
```

## 🔍 Disagreement Analysis

### Analyze Retriever Disagreement

```bash
# Basic disagreement analysis
autorag disagree "your query here"

# Compare specific retrievers
autorag disagree "machine learning" --retrievers bm25 dense

# Use custom corpus
autorag disagree "query" --corpus path/to/corpus.txt

# Save results
autorag disagree "query" --output results.json
```

**Options:**
- `--retrievers`: Retrievers to compare [default: bm25,dense,hybrid]
- `--corpus`: Path to document corpus
- `--k`: Number of documents to retrieve [default: 10]
- `--output`: Output file for results
- `--format`: Output format (`json`, `csv`, `text`) [default: text]

## 🏗️ Configuration Management

### Show Current Configuration

```bash
# Show all configuration
autorag config show

# Show specific config section
autorag config show retrieval

# Show with defaults
autorag config show --defaults
```

### Update Configuration

```bash
# Update from file
autorag config update config/retrieval.yaml

# Update specific values
autorag config set retrieval.dense.model_name all-mpnet-base-v2

# Reset to defaults
autorag config reset retrieval
```

### Validate Configuration

```bash
# Validate all configs
autorag config validate

# Validate specific config
autorag config validate retrieval.yaml
```

## 🔧 Optimization Commands

### Run Optimization

```bash
# Optimize hybrid weights
autorag optimize hybrid

# Optimize with custom parameters
autorag optimize hybrid --grid-size 21 --queries path/to/queries.txt

# Bandit optimization
autorag optimize bandit --num-arms 20 --iterations 100
```

**Options:**
- `--grid-size`: Grid search resolution [default: 11]
- `--queries`: Path to query file
- `--corpus`: Path to corpus file
- `--output`: Output file for optimized config

### A/B Testing

```bash
# Run A/B test
autorag ab-test config_a.yaml config_b.yaml

# With statistical significance
autorag ab-test config_a.yaml config_b.yaml --significance 0.01

# Custom metric
autorag ab-test config_a.yaml config_b.yaml --metric relevance
```

## 📊 Reporting

### Generate Reports

```bash
# Evaluation report
autorag report eval runs/

# Disagreement report
autorag report disagree results.json

# Performance report
autorag report perf benchmark_results.json
```

**Options:**
- `--output`: Output file [default: report.html]
- `--format`: Report format (`html`, `pdf`, `json`) [default: html]
- `--include-plots`: Include plots in report [default: true]

### Export Data

```bash
# Export evaluation results
autorag export eval runs/ results.csv

# Export disagreement analysis
autorag export disagree results.json analysis.csv

# Export time series data
autorag export timeseries tracker.json data.csv
```

## 🗄️ Database Operations

### Index Documents

```bash
# Index documents with BM25
autorag index bm25 path/to/documents/

# Index with dense embeddings
autorag index dense documents/ --model all-MiniLM-L6-v2

# Index to vector database
autorag index qdrant documents/ --url http://localhost:6333
```

**Options:**
- `--collection`: Collection name [default: autorag]
- `--batch-size`: Batch size for indexing [default: 32]
- `--overwrite`: Overwrite existing index [default: false]

### Search Documents

```bash
# Search indexed documents
autorag search "query" --index bm25_index

# Search vector database
autorag search "machine learning" --index qdrant --url http://localhost:6333

# Multi-index search
autorag search "query" --indices bm25,dense,qdrant
```

## 📈 Monitoring

### Start Monitoring

```bash
# Start performance monitoring
autorag monitor start

# Monitor with custom settings
autorag monitor start --interval 60 --output metrics.json
```

### Stop Monitoring

```bash
# Stop monitoring
autorag monitor stop

# Show monitoring status
autorag monitor status
```

### View Metrics

```bash
# Show current metrics
autorag monitor show

# Export metrics
autorag monitor export metrics.csv
```

## 🔧 Development Commands

### Run Tests

```bash
# Run all tests
autorag test

# Run specific test file
autorag test tests/test_retrievers.py

# Run with coverage
autorag test --coverage --output coverage.html
```

### Lint Code

```bash
# Lint all code
autorag lint

# Lint specific files
autorag lint autorag_live/retrievers/

# Fix linting issues
autorag lint --fix
```

### Format Code

```bash
# Format all code
autorag format

# Format specific files
autorag format autorag_live/pipeline/

# Check formatting
autorag format --check
```

## ⚙️ Global Options

### Logging

```bash
# Set log level
autorag --log-level DEBUG command

# Log to file
autorag --log-file logs/autorag.log command

# JSON logging
autorag --json-logs command
```

### Configuration

```bash
# Use custom config directory
autorag --config-dir my_configs/ command

# Override config file
autorag --config retrieval=my_config.yaml command
```

### Performance

```bash
# Set number of threads
autorag --threads 4 command

# Use GPU
autorag --gpu command

# Memory limit
autorag --memory-limit 8GB command
```

## 📋 Examples

### Complete Workflow

```bash
# 1. Set up configuration
autorag config update config/retrieval.yaml

# 2. Index documents
autorag index dense data/documents/

# 3. Run evaluation
autorag eval small --judge-type llm

# 4. Analyze disagreement
autorag disagree "artificial intelligence" --output disagreement.json

# 5. Optimize configuration
autorag optimize hybrid --output optimal_config.yaml

# 6. Generate report
autorag report eval runs/ evaluation_report.html
```

### Automated Optimization

```bash
# Start automated optimization
autorag optimize auto --interval 3600 --max-iterations 50

# Monitor progress
autorag monitor start --output optimization_metrics.json

# Stop after desired time
autorag optimize auto --stop
```

### Production Deployment

```bash
# Validate configuration
autorag config validate

# Run comprehensive tests
autorag test --coverage

# Deploy with monitoring
autorag monitor start &
autorag serve --host 0.0.0.0 --port 8000
```

## 🆘 Troubleshooting

### Common Issues

**Command not found:**
```bash
# Install CLI
pip install autorag-live

# Or run as module
python -m autorag_live.cli command
```

**Configuration errors:**
```bash
# Validate configuration
autorag config validate

# Show current config
autorag config show
```

**Memory issues:**
```bash
# Reduce batch size
autorag --batch-size 8 command

# Use CPU instead of GPU
autorag --no-gpu command
```

**Performance issues:**
```bash
# Enable monitoring
autorag monitor start

# Check metrics
autorag monitor show
```

## 📖 Help System

```bash
# General help
autorag --help

# Command-specific help
autorag eval --help
autorag disagree --help

# Subcommand help
autorag config --help
autorag optimize --help
```
