# Contributing to AutoRAG-Live

Thank you for your interest in contributing to AutoRAG-Live! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## 🤝 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Poetry (for dependency management)
- VS Code (recommended) with Python and Jupyter extensions

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/autorag-live.git
   cd autorag-live
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/rajatsainju2025/autorag-live.git
   ```

## 🛠️ Development Setup

### Install Dependencies

```bash
# Install Python dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Optional: Install development dependencies
poetry install --with dev
```

### Environment Setup

1. Copy configuration files:
   ```bash
   cp config/dev.yaml config/local.yaml
   ```

2. Set up environment variables:
   ```bash
   export AUTORAG_LOG_LEVEL=DEBUG
   # Add other required environment variables
   ```

3. Verify setup:
   ```bash
   poetry run python -c "import autorag_live; print('Setup successful!')"
   ```

### IDE Configuration

#### VS Code

Recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)
- Python Debugger (ms-python.debugpy)
- markdownlint (DavidAnson.vscode-markdownlint)
- GitLens (eamodio.gitlens)
- YAML (redhat.vscode-yaml)

Settings (`.vscode/settings.json`):
```json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.testing.pytestEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  }
}
```

#### PyCharm

1. Open project with Poetry interpreter
2. Enable pre-commit in Settings > Tools > Pre-commit
3. Configure pytest as test runner
4. Enable type checking with mypy plugin

## 📝 Contributing Guidelines

### Types of Contributions

- 🐛 **Bug fixes**: Fix issues in existing code
- ✨ **Features**: Add new functionality
- 📚 **Documentation**: Improve documentation
- 🧪 **Tests**: Add or improve tests
- 🔧 **Maintenance**: Code refactoring, performance improvements
- 🎨 **UI/UX**: Improve user interfaces

### Choosing What to Work On

1. Check [open issues](https://github.com/rajatsainju2025/autorag-live/issues)
2. Look for issues labeled `good first issue` or `help wanted`
3. Comment on the issue to indicate you're working on it
4. Create a draft PR early to get feedback

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Changes

- Write clear, focused commits
- Test your changes thoroughly
- Update documentation as needed
- Follow the code style guidelines

### 3. Run Quality Checks

```bash
# Run all pre-commit hooks
poetry run pre-commit run --all-files

# Run tests
poetry run pytest

# Run type checking
poetry run mypy autorag_live

# Check code coverage
poetry run pytest --cov=autorag_live
```

### 4. Update Documentation

- Update docstrings for new/modified functions
- Update API reference if adding new public APIs
- Update README if adding major features
- Update changelog

## 🧪 Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_specific.py

# Run tests with coverage
poetry run pytest --cov=autorag_live --cov-report=html

# Run quick tests (subset for CI)
poetry run pytest -m "quick"
```

### Writing Tests

- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common setup
- Mock external dependencies
- Aim for high code coverage (>80%)

Example test structure:
```python
import pytest
from autorag_live.retrievers import bm25

class TestBM25Retriever:
    @pytest.fixture
    def sample_corpus(self):
        """Provide sample documents for testing."""
        return ["Document 1", "Document 2", "Document 3"]

    def test_retrieve_basic(self, sample_corpus):
        """Test basic retrieval functionality."""
        retriever = bm25.BM25Retriever(sample_corpus)
        results = retriever.retrieve("query", k=2)
        assert len(results) == 2
        assert all(isinstance(doc, str) for doc in results)

    def test_retrieve_empty_query(self, sample_corpus):
        """Test edge case: empty query."""
        retriever = bm25.BM25Retriever(sample_corpus)
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("", k=2)

    @pytest.mark.parametrize("k,expected", [
        (1, 1),
        (5, 3),  # corpus only has 3 docs
        (0, 0),
    ])
    def test_retrieve_various_k(self, sample_corpus, k, expected):
        """Test retrieval with various k values."""
        retriever = bm25.BM25Retriever(sample_corpus)
        results = retriever.retrieve("test query", k=k)
        assert len(results) == expected
```

### Test Organization

- **Unit tests**: `tests/` with same structure as `autorag_live/`
- **Integration tests**: `tests/integration/`
- **Benchmark tests**: `tests/benchmarks/`
- **Fixtures**: `tests/conftest.py` for shared fixtures

### Continuous Testing

```bash
# Watch mode for continuous testing during development
poetry run pytest-watch

# Run tests on file change with coverage
poetry run ptw -- --cov=autorag_live

# Run only failed tests
poetry run pytest --lf

# Run last failed and then all
poetry run pytest --ff
```

## 📚 Documentation

### Code Documentation

- Use Google-style docstrings
- Document all public functions, classes, and methods
- Include type hints
- Provide usage examples

```python
def retrieve_documents(query: str, corpus: List[str], k: int = 10) -> List[str]:
    """Retrieve top-k relevant documents for a query.

    Uses BM25 scoring to rank documents by relevance to the query.

    Args:
        query: The search query string.
        corpus: List of documents to search through.
        k: Number of top documents to return. Defaults to 10.

    Returns:
        List of top-k relevant document strings.

    Raises:
        ValueError: If k is negative or corpus is empty.

    Example:
        >>> docs = ["Python is great", "Java is also good", "Rust is fast"]
        >>> results = retrieve_documents("Python programming", docs, k=2)
        >>> len(results)
        2
    """
```

### Documentation Updates

- Update `docs/` files for significant changes
- Update README.md for new features
- Update API reference for new public APIs
- Update changelog for user-facing changes

## 🎨 Code Style

### Python Style

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **ruff**: Linting and additional formatting
- **mypy**: Type checking

### Style Guidelines

- Follow PEP 8
- Use type hints for all function parameters and return values
- Use descriptive variable and function names
- Keep functions small and focused
- Use dataclasses for simple data structures
- Write clear, concise comments

### Pre-commit Hooks

Pre-commit hooks automatically check and fix:
- Trailing whitespace
- End-of-file newlines
- Python import sorting
- Code formatting with Black
- Linting with ruff
- Basic syntax errors

## 📤 Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run final checks**:
   ```bash
   poetry run pre-commit run --all-files
   poetry run pytest --cov=autorag_live
   poetry run mypy autorag_live
   ```

3. **Create a pull request**:
   - Use a clear, descriptive title
   - Fill out the PR template
   - Reference related issues
   - Add screenshots for UI changes
   - Link to relevant documentation

4. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes and motivation.

   Fixes #<issue_number>

   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Code refactoring

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed
   - [ ] All tests pass locally
   - [ ] Code coverage maintained/improved

   ## Documentation
   - [ ] Docstrings added/updated
   - [ ] API reference updated
   - [ ] README updated (if needed)
   - [ ] Changelog updated
   - [ ] Migration guide added (for breaking changes)

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Pre-commit hooks pass
   - [ ] No new linting errors
   - [ ] Type hints added
   - [ ] Commits follow conventional commit format
   - [ ] Branch is up to date with main

   ## Performance Impact
   <!-- If applicable, describe performance implications -->
   - Benchmarks:
   - Memory usage:
   - Breaking changes:

   ## Screenshots (if applicable)
   <!-- Add screenshots for UI/UX changes -->
   ```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom retriever plugins
fix: resolve memory leak in dense retriever
docs: update API reference for new metrics
test: add integration tests for CLI commands
refactor: simplify hybrid weight optimization logic
```

### Review Process

- All PRs require review before merging
- Address review comments promptly
- Keep PRs focused on a single concern
- Squash commits when appropriate

## 🌐 Community

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Getting Help

- Check existing issues and documentation first
- Use clear, specific questions
- Provide code examples when possible
- Be patient and respectful

### Recognition

Contributors are recognized through:
- GitHub contributor statistics
- Mention in release notes
- Attribution in documentation

## 🙏 Thank You

Your contributions help make AutoRAG-Live better for everyone. We appreciate your time and effort!
