"""Tests for observability metrics."""

import pytest

from autorag_live.core.metrics_export import CacheMetric, LatencyMetric, MetricsExporter


class TestLatencyMetric:
    """Tests for LatencyMetric dataclass."""

    def test_creation(self):
        """Test creating a latency metric."""
        metric = LatencyMetric("query", 100.5)
        assert metric.operation == "query"
        assert metric.duration_ms == 100.5
        assert metric.timestamp > 0

    def test_with_tags(self):
        """Test latency metric with tags."""
        metric = LatencyMetric("search", 50.0, tags={"model": "bge", "status": "ok"})
        assert metric.tags["model"] == "bge"
        assert metric.tags["status"] == "ok"


class TestCacheMetric:
    """Tests for CacheMetric dataclass."""

    def test_hit_rate_empty(self):
        """Test hit rate with no hits/misses."""
        metric = CacheMetric("query_cache")
        assert metric.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metric = CacheMetric("embedding_cache", hits=80, misses=20)
        assert metric.hit_rate == 0.8

    def test_hit_rate_all_hits(self):
        """Test hit rate when all are hits."""
        metric = CacheMetric("token_cache", hits=100, misses=0)
        assert metric.hit_rate == 1.0


class TestMetricsExporter:
    """Tests for MetricsExporter."""

    def test_init(self):
        """Test initialization."""
        exporter = MetricsExporter("test-service")
        assert exporter.service_name == "test-service"
        assert len(exporter.latency_metrics) == 0
        assert len(exporter.cache_metrics) == 0

    def test_record_latency(self):
        """Test recording latency."""
        exporter = MetricsExporter()
        exporter.record_latency("retrieve", 250.5)
        assert len(exporter.latency_metrics) == 1
        assert exporter.latency_metrics[0].operation == "retrieve"
        assert exporter.latency_metrics[0].duration_ms == 250.5

    def test_record_latency_with_tags(self):
        """Test recording latency with tags."""
        exporter = MetricsExporter()
        exporter.record_latency("rerank", 100.0, tags={"model": "bge-reranker", "k": "10"})
        metric = exporter.latency_metrics[0]
        assert metric.tags["model"] == "bge-reranker"
        assert metric.tags["k"] == "10"

    def test_cache_hit(self):
        """Test recording cache hit."""
        exporter = MetricsExporter()
        exporter.record_cache_hit("embedding_cache")
        exporter.record_cache_hit("embedding_cache")
        assert exporter.cache_metrics["embedding_cache"].hits == 2

    def test_cache_miss(self):
        """Test recording cache miss."""
        exporter = MetricsExporter()
        exporter.record_cache_miss("query_cache")
        assert exporter.cache_metrics["query_cache"].misses == 1

    def test_cache_hit_rate(self):
        """Test getting cache hit rate."""
        exporter = MetricsExporter()
        exporter.record_cache_hit("doc_cache")
        exporter.record_cache_hit("doc_cache")
        exporter.record_cache_miss("doc_cache")
        rate = exporter.get_cache_hit_rate("doc_cache")
        assert rate == pytest.approx(2.0 / 3.0)

    def test_cache_hit_rate_nonexistent(self):
        """Test hit rate for nonexistent cache."""
        exporter = MetricsExporter()
        rate = exporter.get_cache_hit_rate("nonexistent")
        assert rate == 0.0

    def test_export_prometheus_format(self):
        """Test prometheus format export."""
        exporter = MetricsExporter()
        exporter.record_latency("retrieve", 100.0, tags={"status": "success"})
        exporter.record_cache_hit("cache1")
        exporter.record_cache_miss("cache1")

        prometheus_str = exporter.export_prometheus_format()
        assert "autorag_operation_latency_ms" in prometheus_str
        assert "autorag_cache_hits" in prometheus_str
        assert "autorag_cache_misses" in prometheus_str
        assert "autorag_cache_hit_rate" in prometheus_str

    def test_export_dict(self):
        """Test dictionary export."""
        exporter = MetricsExporter("my-service")
        exporter.record_latency("search", 50.0)
        exporter.record_cache_hit("results")

        export_dict = exporter.export_dict()
        assert export_dict["service_name"] == "my-service"
        assert len(export_dict["latency_metrics"]) == 1
        assert "results" in export_dict["cache_metrics"]

    def test_clear_metrics(self):
        """Test clearing metrics."""
        exporter = MetricsExporter()
        exporter.record_latency("op", 100.0)
        exporter.record_cache_hit("cache")

        assert len(exporter.latency_metrics) > 0
        assert len(exporter.cache_metrics) > 0

        exporter.clear()
        assert len(exporter.latency_metrics) == 0
        assert len(exporter.cache_metrics) == 0
