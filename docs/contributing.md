# Contributing

Welcome! We appreciate your interest in contributing to AutoRAG-Live. This document provides guidelines for contributors.

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/autorag-live.git
   cd autorag-live
   ```

2. **Set up Development Environment**
   ```bash
   # Install with development dependencies
   pip install -e .[dev]

   # Or use Poetry
   poetry install
   ```

3. **Run Tests**
   ```bash
   pytest
   ```

4. **Check Code Quality**
   ```bash
   # Lint code
   ruff check .

   # Format code
   ruff format .

   # Type check
   mypy autorag_live
   ```

## 📝 Development Workflow

### 1. Choose an Issue

- Check [GitHub Issues](https://github.com/rajatsainju2025/autorag-live/issues) for open tasks
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or fix branch
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clear, focused commits
- Follow the [commit message conventions](#commit-conventions)
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=autorag_live

# Run specific tests
pytest tests/retrievers/test_bm25.py
```

### 5. Submit a Pull Request

- Push your branch to GitHub
- Create a Pull Request with a clear description
- Reference any related issues
- Wait for review and address feedback

## 🐛 Reporting Bugs

### Bug Report Template

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**:
   ```python
   # Minimal code example that reproduces the issue
   from autorag_live.retrievers import bm25
   corpus = ["test document"]
   results = bm25.bm25_retrieve("query", corpus, k=1)
   # Expected: ["test document"]
   # Actual: []
   ```
3. **Environment**:
   - Python version: 3.11
   - OS: macOS 13.0
   - AutoRAG-Live version: 0.1.0
4. **Expected vs Actual Behavior**
5. **Additional Context**: Screenshots, logs, etc.

## ✨ Feature Requests

### Feature Request Template

For new features, please provide:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: High-level description of the solution
3. **Alternatives Considered**: Other approaches you considered
4. **Use Cases**: Specific examples of how this would be used
5. **Implementation Notes**: Any technical considerations

## 🏗️ Architecture Guidelines

### Code Organization

```
autorag_live/
├── retrievers/          # Retrieval algorithms
├── evals/              # Evaluation metrics and judges
├── pipeline/           # Optimization and pipeline logic
├── disagreement/       # Disagreement analysis
├── augment/            # Data augmentation
├── data/               # Data structures and time series
├── utils/              # Utilities and configuration
└── types/              # Type definitions
```

### Design Principles

- **Modularity**: Components should be loosely coupled
- **Extensibility**: Easy to add new retrievers, metrics, etc.
- **Type Safety**: Use type hints throughout
- **Testability**: Write testable code with clear interfaces
- **Documentation**: Document all public APIs

## 📚 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# ✅ Good
def calculate_score(query: str, documents: List[str]) -> float:
    """Calculate relevance score for query-document pairs."""
    if not query or not documents:
        return 0.0

    scores = []
    for doc in documents:
        score = len(set(query.split()) & set(doc.split()))
        scores.append(score)

    return max(scores) if scores else 0.0

# ❌ Bad
def calc_score(q,d):
    if not q or not d:
        return 0
    s=[]
    for doc in d:
        s.append(len(set(q.split())&set(doc.split())))
    return max(s)if s else 0
```

### Type Hints

```python
# ✅ Use descriptive types
from typing import List, Dict, Optional, Union, Any
from autorag_live.types import RetrieverProtocol

def retrieve_documents(
    query: str,
    corpus: List[str],
    k: int = 10,
    options: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Retrieve top-k relevant documents."""
    pass

# ✅ Use generics for complex types
from typing import TypeVar, Generic

T = TypeVar('T')
class Result(Generic[T]):
    def __init__(self, value: T, score: float):
        self.value = value
        self.score = score
```

### Error Handling

```python
# ✅ Explicit error handling
def safe_retrieve(query: str, corpus: List[str], k: int) -> List[str]:
    """Safely retrieve documents with error handling."""
    try:
        if not query:
            raise ValueError("Query cannot be empty")
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        if k <= 0:
            raise ValueError("k must be positive")

        return retrieve_documents(query, corpus, k)

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

# ✅ Custom exceptions for domain errors
class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass

class ConfigurationError(RetrievalError):
    """Exception for configuration errors."""
    pass
```

## 🧪 Testing Guidelines

### Test Coverage

- Aim for 90%+ code coverage
- Test both success and failure cases
- Use parametrized tests for multiple inputs

```python
import pytest

class TestBM25Retriever:
    @pytest.mark.parametrize("query,corpus,k,expected_length", [
        ("test", ["test doc"], 1, 1),
        ("missing", ["other doc"], 1, 0),
        ("", ["doc"], 1, 0),
    ])
    def test_bm25_retrieve(self, query, corpus, k, expected_length):
        results = bm25.bm25_retrieve(query, corpus, k)
        assert len(results) == expected_length
```

### Mocking

```python
from unittest.mock import Mock, patch

def test_with_mocking():
    """Test with mocked dependencies."""
    with patch('autorag_live.retrievers.dense.sentence_transformers') as mock_st:
        mock_model = Mock()
        mock_st.SentenceTransformer.return_value = mock_model
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        results = dense.dense_retrieve("query", ["doc"], k=1)
        assert len(results) == 1
```

## 📖 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def hybrid_retrieve(
    query: str,
    corpus: List[str],
    k: int = 10,
    weights: Optional[HybridWeights] = None
) -> List[str]:
    """Retrieve documents using hybrid BM25 + dense approach.

    Combines BM25 lexical matching with dense semantic embeddings
    for improved retrieval performance.

    Args:
        query: Search query string.
        corpus: List of documents to search.
        k: Number of documents to retrieve. Defaults to 10.
        weights: Custom weights for BM25 vs dense. If None, uses
            balanced weights (0.5, 0.5).

    Returns:
        List of top-k relevant documents ranked by combined score.

    Raises:
        ValueError: If query is empty or corpus is empty.

    Example:
        >>> corpus = ["AI is artificial intelligence", "ML is machine learning"]
        >>> results = hybrid_retrieve("artificial intelligence", corpus, k=1)
        >>> print(results[0])
        'AI is artificial intelligence'
    """
```

### Documentation Updates

- Update docstrings when changing function signatures
- Add examples for complex functionality
- Keep README and docs in sync

## 🔄 Commit Conventions

We follow [Conventional Commits](https://conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

### Types

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

### Examples

```bash
# Feature commit
feat(retrievers): add FAISS vector database support

- Implement FAISSRetriever class
- Add GPU acceleration support
- Include comprehensive tests

# Fix commit
fix(evals): handle empty results in exact_match metric

Previously, exact_match would fail with empty predictions.
Now returns 0.0 for empty inputs.

# Documentation commit
docs(readme): update installation instructions

Add Docker installation method and troubleshooting section.
```

## 🔧 Tools and Configuration

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### IDE Configuration

Recommended VS Code settings:

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.typeChecking.mode": "strict",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## 🤝 Code Review Process

### Review Checklist

**For Reviewers:**
- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes without discussion
- [ ] Performance implications considered

**For Contributors:**
- [ ] Self-review your code
- [ ] Write clear commit messages
- [ ] Add tests for new functionality
- [ ] Update documentation
- [ ] Consider edge cases and error handling

### Review Comments

```markdown
<!-- ✅ Good review comment -->
The error handling here could be more specific. Instead of catching
generic Exception, consider catching ValueError for invalid inputs
and ConnectionError for network issues.

<!-- ❌ Bad review comment -->
This is wrong.
```

## 🎉 Recognition

Contributors are recognized through:

- GitHub contributor statistics
- Attribution in CHANGELOG.md
- Mention in release notes
- Community recognition

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/rajatsainju2025/autorag-live/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rajatsainju2025/autorag-live/discussions)
- **Discord**: Join our community Discord server

## 📋 License

By contributing to AutoRAG-Live, you agree that your contributions will be licensed under the MIT License.