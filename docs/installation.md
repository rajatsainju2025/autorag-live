# Installation

This page covers how to install AutoRAG-Live for different use cases.

## 🚀 Quick Install

### From PyPI (Recommended)

```bash
pip install autorag-live
```

### From Source (Development)

```bash
git clone https://github.com/rajatsainju2025/autorag-live
cd autorag-live
pip install -e .
```

## 📦 Optional Dependencies

AutoRAG-Live supports various vector databases and search engines. Install optional dependencies as needed:

### FAISS Support

```bash
pip install autorag-live[faiss]
# or
pip install faiss-cpu
```

### Qdrant Support

```bash
pip install autorag-live[qdrant]
# or
pip install qdrant-client
```

### Elasticsearch Support

```bash
pip install autorag-live[elasticsearch]
# or
pip install elasticsearch
```

## 🛠️ Development Setup

⚠️ **Important**: Ensure you have Python 3.10+ installed before proceeding.

For contributors and development:

```bash
# Check Python version first
python --version  # Must be 3.10.0 or higher

git clone https://github.com/rajatsainju2025/autorag-live
cd autorag-live

# Install with all optional dependencies
pip install -e .[faiss,qdrant,elasticsearch]

# Install development dependencies
pip install -e .[dev]
```

### Development Tools

The development setup includes:

- **pytest**: Testing framework
- **mypy**: Type checking
- **ruff**: Linting and formatting
- **pre-commit**: Git hooks
- **mkdocs**: Documentation
- **mkdocstrings**: API documentation generation

## 🔧 System Requirements

### Minimum Requirements

- Python 3.10+
- 4GB RAM
- 2GB disk space

### Recommended Requirements

- Python 3.11+
- 8GB RAM
- GPU with CUDA support (for dense retrieval)
- 10GB disk space

## 🐳 Docker

Build and run with Docker:

```bash
# Build the image
docker build -t autorag-live .

# Run the container
docker run -it autorag-live
```

## ✅ Verification

Verify your installation:

```python
import autorag_live
print(autorag_live.__version__)

# Test basic functionality
from autorag_live.retrievers import bm25
results = bm25.bm25_retrieve("test query", ["test document"], k=1)
print(results)
```

## 🆘 Troubleshooting

### Common Issues

**Import Error: No module named 'sentence_transformers'**

```bash
pip install sentence-transformers
```

**CUDA out of memory**

Reduce batch size in configuration or use CPU-only mode:

```yaml
# config/retrieval.yaml
dense:
  device: cpu
  batch_size: 8
```

**Elasticsearch connection failed**

Ensure Elasticsearch is running:

```bash
# Start Elasticsearch with Docker
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0
```