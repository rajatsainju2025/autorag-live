# Changelog

All notable changes to AutoRAG-Live will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized logging configuration with structured logging
- Comprehensive error handling and logging across all modules
- MkDocs documentation with API reference generation
- Expanded test coverage for all retrievers (Qdrant, BM25, dense, FAISS, hybrid, Elasticsearch)
- Type system with comprehensive type hints and custom exceptions
- Configuration management with YAML files and ConfigManager singleton
- Retriever base class with common interface and type hints
- Project configuration with Poetry, mypy, ruff, and pre-commit
- CLI interface with evaluation, disagreement analysis, and optimization commands

### Changed
- Refactored all modules to use centralized logging
- Improved modularity with better separation of concerns
- Enhanced error handling with custom exceptions and logging
- Updated project structure for better maintainability

### Fixed
- Resolved duplicate sections in pyproject.toml
- Fixed dataclass mutable default issues
- Corrected import and mocking issues in tests
- Fixed JSON serialization errors with mock objects
- Resolved numpy array handling in tests

## [0.1.0] - 2024-01-XX

### Added
- Initial release of AutoRAG-Live
- Core retrieval algorithms: BM25, dense, hybrid
- Vector database support: Qdrant, FAISS, Elasticsearch
- Evaluation metrics: exact match, F1, BLEU, ROUGE
- Disagreement analysis with Jaccard and Kendall tau
- LLM-based evaluation judges
- Optimization pipeline with bandit algorithms
- Basic CLI interface
- Comprehensive test suite

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

---

## Types of Changes

- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities

## Versioning

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

## Contributing

See [Contributing Guide](contributing.md) for information about contributing to AutoRAG-Live.

## Migration Guide

### From 0.0.x to 0.1.0

#### Configuration Changes

The configuration system has been completely rewritten. Update your configuration files:

```yaml
# Old format (if any)
# ... old config ...

# New format
retrieval:
  bm25:
    k1: 1.5
    b: 0.75
  dense:
    model_name: "all-MiniLM-L6-v2"
    device: "auto"

evaluation:
  judge_type: "deterministic"
  metrics: ["exact_match", "f1", "relevance"]
```

#### API Changes

- All retriever functions now return typed results
- Error handling is more explicit with custom exceptions
- Logging is now centralized - remove custom logging setup

```python
# Old code
from autorag_live.retrievers import bm25
results = bm25.bm25_retrieve("query", corpus)  # May have untyped results

# New code
from autorag_live.retrievers import bm25
from typing import List
results: List[str] = bm25.bm25_retrieve("query", corpus, k=10)
```

#### Dependency Changes

New required dependencies:
- `sentence-transformers` for dense retrieval
- `pyyaml` for configuration
- `omegaconf` and `hydra-core` for config management

Optional dependencies for vector databases:
- `qdrant-client` for Qdrant support
- `faiss-cpu` for FAISS support
- `elasticsearch` for Elasticsearch support

## Acknowledgments

Thanks to all contributors who helped make AutoRAG-Live possible:

- [Raj Atul Sainju](https://github.com/rajatsainju2025) - Project creator and maintainer

## Contact

- **Issues**: [GitHub Issues](https://github.com/rajatsainju2025/autorag-live/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rajatsainju2025/autorag-live/discussions)
- **Email**: rajatsainju2025@gmail.com
