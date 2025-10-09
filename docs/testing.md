# Testing

Comprehensive testing strategies for AutoRAG-Live development and validation.

## 🧪 Test Types

### Unit Tests

Test individual components in isolation.

```python
# tests/retrievers/test_bm25.py
import pytest
from autorag_live.retrievers import bm25

class TestBM25Retriever:
    def test_bm25_retrieve_basic(self):
        corpus = ["hello world", "foo bar", "hello foo"]
        results = bm25.bm25_retrieve("hello", corpus, k=2)
        assert len(results) == 2
        assert "hello world" in results

    def test_bm25_retrieve_empty_query(self):
        corpus = ["test document"]
        results = bm25.bm25_retrieve("", corpus, k=1)
        assert results == []

    @pytest.mark.parametrize("k", [1, 2, 5])
    def test_bm25_retrieve_different_k(self, k):
        corpus = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        results = bm25.bm25_retrieve("doc", corpus, k=k)
        assert len(results) == k
```

### Integration Tests

Test component interactions.

```python
# tests/integration/test_retrieval_pipeline.py
import pytest
from autorag_live.retrievers import hybrid
from autorag_live.evals import small

def test_hybrid_retrieval_integration():
    """Test complete retrieval pipeline."""
    corpus = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing handles text"
    ]

    query = "artificial intelligence"
    results = hybrid.hybrid_retrieve(query, corpus, k=2)

    assert len(results) == 2
    assert all(isinstance(doc, str) for doc in results)

def test_evaluation_integration():
    """Test evaluation with retrieval."""
    results = small.run_small_suite(runs_dir="test_runs")
    assert "metrics" in results
    assert "f1" in results["metrics"]
    assert 0 <= results["metrics"]["f1"] <= 1
```

### End-to-End Tests

Test complete workflows.

```python
# tests/e2e/test_optimization_loop.py
import pytest
from autorag_live.pipeline import acceptance_policy, hybrid_optimizer

def test_optimization_loop_e2e():
    """Test complete optimization workflow."""
    # Create test data
    corpus = ["test document {}".format(i) for i in range(10)]
    queries = ["query {}".format(i) for i in range(5)]

    # Optimize hybrid weights
    weights, score = hybrid_optimizer.grid_search_hybrid_weights(
        queries, corpus, k=3
    )

    assert 0 <= weights.bm25_weight <= 1
    assert 0 <= weights.dense_weight <= 1
    assert abs(weights.bm25_weight + weights.dense_weight - 1.0) < 0.01
    assert 0 <= score <= 1

    # Test acceptance policy
    policy = acceptance_policy.AcceptancePolicy(threshold=0.01)
    # Mock evaluation would go here
```

## 🛠️ Testing Tools

### Pytest Configuration

```python
# pytest.ini or pyproject.toml
[tool:pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=autorag_live",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "e2e: marks tests as end-to-end tests",
    "gpu: marks tests that require GPU"
]
```

### Fixtures

Common test fixtures.

```python
# tests/conftest.py
import pytest
from autorag_live.retrievers import bm25, dense, hybrid

@pytest.fixture
def sample_corpus():
    """Sample document corpus for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Python is a great programming language",
        "Retrieval augmented generation combines retrieval and generation",
        "Natural language processing is a field of AI"
    ]

@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "machine learning",
        "programming languages",
        "artificial intelligence"
    ]

@pytest.fixture
def mock_retrievers(sample_corpus):
    """Mock retrievers for testing."""
    # Mock expensive operations
    return {
        "bm25": lambda q, k=5: bm25.bm25_retrieve(q, sample_corpus, k),
        "dense": lambda q, k=5: dense.dense_retrieve(q, sample_corpus, k),
        "hybrid": lambda q, k=5: hybrid.hybrid_retrieve(q, sample_corpus, k)
    }
```

### Mocking External Dependencies

```python
# tests/retrievers/test_qdrant_adapter.py
import pytest
from unittest.mock import Mock, patch
from autorag_live.retrievers import QdrantRetriever

class TestQdrantRetriever:
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_qdrant_retriever_init(self, mock_client):
        """Test Qdrant retriever initialization."""
        mock_client.return_value = Mock()

        retriever = QdrantRetriever(
            url="http://localhost:6333",
            collection_name="test"
        )

        assert retriever.collection_name == "test"
        mock_client.assert_called_once_with(url="http://localhost:6333")

    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_qdrant_add_documents(self, mock_client):
        """Test adding documents to Qdrant."""
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        retriever = QdrantRetriever()
        documents = ["doc1", "doc2", "doc3"]

        retriever.add_documents(documents)

        # Verify upsert was called
        mock_instance.upsert.assert_called_once()
        call_args = mock_instance.upsert.call_args
        assert len(call_args[1]['points']) == 3
```

## 📊 Coverage Testing

### Coverage Configuration

```python
# pyproject.toml
[tool.coverage.run]
source = ["autorag_live"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/__pycache__/*",
    "autorag_live/_version.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

### Coverage Goals

- **Unit Tests**: 90%+ coverage
- **Integration Tests**: Cover all major workflows
- **E2E Tests**: Cover complete user journeys

## 🚀 Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/retrievers/test_bm25.py

# Run specific test
pytest tests/retrievers/test_bm25.py::TestBM25Retriever::test_bm25_retrieve_basic

# Run tests matching pattern
pytest -k "bm25 and retrieve"
```

### Advanced Options

```bash
# Run with coverage
pytest --cov=autorag_live --cov-report=html

# Run slow tests only
pytest -m slow

# Run tests in parallel
pytest -n auto

# Stop on first failure
pytest -x

# Verbose output
pytest -v

# Debug failing test
pytest --pdb
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: |
        pytest --cov=autorag_live --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 🐛 Debugging Tests

### Common Debugging Techniques

```python
# Add debug prints
def test_problematic_function():
    result = some_function()
    print(f"Debug: result = {result}")  # Temporary debug
    assert result == expected

# Use pytest fixtures for setup
@pytest.fixture
def debug_setup():
    # Setup code
    yield
    # Teardown code

# Parameterize tests
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("123", "123")
])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

### Debugging Tools

```python
# pytest debugging
pytest --pdb  # Drop into debugger on failure
pytest -s     # Don't capture output
pytest -v     # Verbose output

# Coverage debugging
pytest --cov=autorag_live --cov-report=html
# Open htmlcov/index.html

# Profiling tests
pytest --durations=10  # Show slowest 10 tests
```

## 📈 Performance Testing

### Benchmark Tests

```python
# tests/performance/test_retrieval_speed.py
import pytest
import time
from autorag_live.retrievers import bm25, dense

class TestRetrievalPerformance:
    @pytest.fixture
    def large_corpus(self):
        """Generate large test corpus."""
        return [f"Document number {i} with some content" for i in range(1000)]

    def test_bm25_performance(self, large_corpus, benchmark):
        """Benchmark BM25 retrieval performance."""
        query = "document content"

        # Use pytest-benchmark
        result = benchmark(
            bm25.bm25_retrieve,
            query=query,
            corpus=large_corpus,
            k=10
        )

        assert len(result) == 10

    def test_dense_performance(self, large_corpus, benchmark):
        """Benchmark dense retrieval performance."""
        query = "machine learning"

        result = benchmark(
            dense.dense_retrieve,
            query=query,
            corpus=large_corpus,
            k=10
        )

        assert len(result) == 10
```

### Memory Testing

```python
# tests/performance/test_memory_usage.py
import pytest
import psutil
import os

def test_memory_usage():
    """Test memory usage doesn't exceed limits."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Run memory-intensive operation
    large_corpus = ["large document"] * 10000
    results = bm25.bm25_retrieve("query", large_corpus, k=100)

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # Assert memory increase is reasonable
    assert memory_increase < 500  # Less than 500MB increase
```

## 🔧 Test Maintenance

### Test Organization

```
tests/
├── unit/                    # Unit tests
│   ├── retrievers/
│   ├── evals/
│   └── utils/
├── integration/            # Integration tests
├── e2e/                    # End-to-end tests
├── performance/            # Performance tests
├── conftest.py            # Shared fixtures
└── test_*.py             # General tests
```

### Test Quality Checks

```python
# tests/test_quality.py
def test_all_public_functions_have_tests():
    """Ensure all public functions are tested."""
    import autorag_live
    import inspect

    # Get all public functions
    public_functions = []
    for name in dir(autorag_live):
        if not name.startswith('_'):
            obj = getattr(autorag_live, name)
            if callable(obj):
                public_functions.append(name)

    # Check tests exist (simplified)
    # This would need more sophisticated implementation
    pass

def test_test_coverage_meets_threshold():
    """Ensure test coverage meets minimum threshold."""
    # This would integrate with coverage tools
    pass
```

## 🎯 Best Practices

### Writing Good Tests

```python
# ✅ Good: Clear, focused test
def test_bm25_returns_correct_number_of_results():
    corpus = ["doc1", "doc2", "doc3"]
    results = bm25.bm25_retrieve("query", corpus, k=2)
    assert len(results) == 2

# ❌ Bad: Tests too many things
def test_everything():
    # This test does too much
    pass

# ✅ Good: Use descriptive names
def test_retrieval_with_empty_corpus_returns_empty_list():
    results = bm25.bm25_retrieve("query", [], k=5)
    assert results == []

# ✅ Good: Test edge cases
@pytest.mark.parametrize("query,corpus,expected", [
    ("", ["doc"], []),
    ("query", [], []),
    ("match", ["match", "no match"], ["match"])
])
def test_bm25_edge_cases(query, corpus, expected):
    results = bm25.bm25_retrieve(query, corpus, k=10)
    assert results == expected
```

### Test Isolation

```python
# ✅ Good: Isolated test with cleanup
def test_temporary_file_creation(tmp_path):
    """Test that uses temporary directory."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value")

    # Test logic here
    assert config_file.exists()

    # Cleanup automatic with tmp_path

# ✅ Good: Mock external dependencies
@patch('autorag_live.utils.requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {"status": "ok"}

    result = call_external_api()
    assert result["status"] == "ok"
```

## 📊 Test Reporting

### Generate Test Reports

```bash
# HTML coverage report
pytest --cov=autorag_live --cov-report=html

# XML coverage for CI
pytest --cov=autorag_live --cov-report=xml

# JUnit XML for CI
pytest --junitxml=test-results.xml

# Performance profiling
pytest --durations=10
```

### Custom Reporting

```python
# tests/conftest.py
def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")

def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip slow tests unless requested
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
```

## 🔬 Advanced Testing Techniques

### Property-Based Testing

Use hypothesis for testing with generated inputs.

```python
# tests/test_property_based.py
import pytest
from hypothesis import given, strategies as st
from autorag_live.retrievers import bm25

@given(
    corpus=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=50),
    query=st.text(min_size=1, max_size=50),
    k=st.integers(min_value=1, max_value=10)
)
def test_bm25_retrieve_properties(corpus, query, k):
    """Property-based test for BM25 retrieval."""
    k = min(k, len(corpus))  # Ensure k doesn't exceed corpus size

    results = bm25.bm25_retrieve(query, corpus, k=k)

    # Properties that should always hold
    assert len(results) <= k
    assert len(results) <= len(corpus)
    assert all(doc in corpus for doc in results)

@given(st.lists(st.text(), min_size=1))
def test_retrieval_consistency(corpus):
    """Test that retrieval is deterministic."""
    query = "test query"
    k = min(5, len(corpus))

    results1 = bm25.bm25_retrieve(query, corpus, k=k)
    results2 = bm25.bm25_retrieve(query, corpus, k=k)

    assert results1 == results2
```

### Performance Testing

Test performance characteristics and benchmarks.

```python
# tests/performance/test_retrieval_performance.py
import pytest
import time
from autorag_live.utils.performance import PerformanceMonitor

class TestRetrievalPerformance:
    @pytest.fixture
    def large_corpus(self):
        """Generate a large test corpus."""
        return [f"Document number {i} with some content" for i in range(1000)]

    def test_bm25_performance_scaling(self, large_corpus, benchmark):
        """Test BM25 performance with different corpus sizes."""
        def bm25_search():
            return bm25.bm25_retrieve "test query", large_corpus, k=10

        # Benchmark the function
        result = benchmark(bm25_search)

        assert len(result) == 10
        # Check that performance is reasonable
        assert benchmark.stats.mean < 1.0  # Should complete in < 1 second

    def test_memory_usage_bounds(self, large_corpus):
        """Test memory usage stays within bounds."""
        monitor = PerformanceMonitor()

        with monitor.memory_monitor():
            results = bm25.bm25_retrieve("query", large_corpus, k=50)

        memory_mb = monitor.current_memory_usage
        assert memory_mb < 500  # Less than 500MB for this operation

    @pytest.mark.parametrize("corpus_size", [100, 1000, 10000])
    def test_scalability(self, corpus_size):
        """Test how performance scales with corpus size."""
        corpus = [f"doc {i}" for i in range(corpus_size)]

        start_time = time.time()
        results = bm25.bm25_retrieve("test", corpus, k=10)
        duration = time.time() - start_time

        # Performance should scale roughly linearly
        # (allowing some overhead for larger data structures)
        assert duration < corpus_size / 1000  # Rough heuristic
```

### Load Testing

Test system behavior under load.

```python
# tests/load/test_concurrent_retrieval.py
import pytest
import asyncio
import concurrent.futures
from autorag_live.retrievers import bm25

class TestConcurrentRetrieval:
    @pytest.fixture
    def corpus(self):
        return [f"Document {i} with searchable content" for i in range(100)]

    def test_concurrent_queries(self, corpus):
        """Test handling multiple concurrent queries."""
        queries = [f"query {i}" for i in range(10)]

        def run_query(query):
            return bm25.bm25_retrieve(query, corpus, k=5)

        # Run queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_query, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10
        assert all(len(r) == 5 for r in results)

    @pytest.mark.asyncio
    async def test_async_retrieval(self, corpus):
        """Test async retrieval patterns."""
        async def async_retrieve(query):
            # Simulate async operation
            await asyncio.sleep(0.01)
            return bm25.bm25_retrieve(query, corpus, k=3)

        queries = [f"async query {i}" for i in range(5)]
        tasks = [async_retrieve(q) for q in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(len(r) == 3 for r in results)
```

### Fuzz Testing

Test with random or malformed inputs.

```python
# tests/fuzz/test_input_fuzzing.py
import pytest
from autorag_live.retrievers import bm25

class TestInputFuzzing:
    @pytest.mark.parametrize("malformed_input", [
        "",  # Empty string
        "a" * 10000,  # Very long string
        "🚀🔥💯",  # Unicode emojis
        "<script>alert('xss')</script>",  # Potentially dangerous input
        "query\nwith\nnewlines",  # Multi-line input
        "query\twith\ttabs",  # Tab characters
        "query\x00with\x00nulls",  # Null bytes
    ])
    def test_malformed_queries(self, malformed_input):
        """Test retrieval with various malformed inputs."""
        corpus = ["normal document", "another document"]

        # Should not crash
        results = bm25.bm25_retrieve(malformed_input, corpus, k=2)

        # Should return some results or empty list
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_extreme_corpus_sizes(self):
        """Test with extreme corpus sizes."""
        # Empty corpus
        results = bm25.bm25_retrieve("query", [], k=1)
        assert results == []

        # Single document corpus
        results = bm25.bm25_retrieve("query", ["single doc"], k=5)
        assert len(results) == 1

        # Very large k
        corpus = ["doc1", "doc2"]
        results = bm25.bm25_retrieve("query", corpus, k=100)
        assert len(results) == 2  # Should not exceed corpus size
```

### Integration Testing with External Services

Test interactions with external dependencies.

```python
# tests/integration/test_external_services.py
import pytest
from unittest.mock import Mock, patch
from autorag_live.retrievers.elasticsearch_adapter import ElasticsearchRetriever

class TestElasticsearchIntegration:
    @patch('autorag_live.retrievers.elasticsearch_adapter.Elasticsearch')
    def test_elasticsearch_connection(self, mock_es):
        """Test Elasticsearch connection handling."""
        # Mock successful connection
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.search.return_value = {
            'hits': {'hits': [{'_source': {'content': 'test doc'}}]}
        }
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever(host="localhost", port=9200)
        results = retriever.retrieve("test query", index_name="test", k=1)

        assert len(results) == 1
        assert results[0] == "test doc"
        mock_client.search.assert_called_once()

    @patch('autorag_live.retrievers.elasticsearch_adapter.Elasticsearch')
    def test_elasticsearch_connection_failure(self, mock_es):
        """Test graceful handling of connection failures."""
        mock_es.side_effect = ConnectionError("Connection refused")

        with pytest.raises(ConnectionError):
            ElasticsearchRetriever(host="invalid-host", port=9200)
```

### CI/CD Testing

Ensure tests work in automated environments.

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install

    - name: Run tests
      run: |
        poetry run pytest --cov=autorag_live --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Data Management

Manage test data effectively.

```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_corpus():
    """Provide a standard test corpus."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Natural language processing with transformers",
        "Retrieval augmented generation combines retrieval and generation",
        "Vector databases store high-dimensional embeddings"
    ]

@pytest.fixture
def mock_embeddings():
    """Provide mock embeddings for testing."""
    import numpy as np
    np.random.seed(42)  # For reproducible tests
    return np.random.randn(5, 384)  # 5 docs, 384 dimensions
```
