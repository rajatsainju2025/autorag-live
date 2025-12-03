"""Edge case tests for augmentation modules."""


from autorag_live.augment.hard_negatives import sample_hard_negatives
from autorag_live.augment.query_rewrites import rewrite_query


class TestQueryRewritesEdgeCases:
    """Test query rewrite function with edge cases."""

    def test_rewrite_empty_query(self):
        """Test rewriting empty query."""
        result = rewrite_query("")
        assert isinstance(result, list)
        assert len(result) >= 0  # Should handle gracefully

    def test_rewrite_single_word(self):
        """Test rewriting single word query."""
        result = rewrite_query("test")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_rewrite_very_long_query(self):
        """Test rewriting very long query."""
        long_query = " ".join(["word"] * 100)
        result = rewrite_query(long_query)
        assert isinstance(result, list)

    def test_rewrite_special_characters(self):
        """Test rewriting query with special characters."""
        result = rewrite_query("test@#$%query!")
        assert isinstance(result, list)

    def test_rewrite_unicode_query(self):
        """Test rewriting query with unicode characters."""
        result = rewrite_query("测试查询 тест запрос")
        assert isinstance(result, list)


class TestHardNegativesEdgeCases:
    """Test hard negatives sampling with edge cases."""

    def test_sample_empty_pools(self):
        """Test sampling with empty negative pool."""
        result = sample_hard_negatives(["doc1"], [], num_negatives=5)
        assert result == []

    def test_sample_zero_negatives(self):
        """Test sampling with num_negatives=0."""
        positive = ["doc1"]
        negative_pool = [["doc2", "doc3"]]
        result = sample_hard_negatives(positive, negative_pool, num_negatives=0)
        assert len(result) == 0

    def test_sample_more_negatives_than_available(self):
        """Test sampling with num_negatives > available negatives."""
        positive = ["doc1"]
        negative_pool = [["doc2", "doc3"]]
        result = sample_hard_negatives(positive, negative_pool, num_negatives=10)
        assert len(result) <= 2  # Only 2 negatives available

    def test_sample_overlapping_documents(self):
        """Test sampling with overlapping positive and negative docs."""
        positive = ["doc1", "doc2"]
        negative_pool = [["doc2", "doc3", "doc4"]]
        result = sample_hard_negatives(positive, negative_pool, num_negatives=5)
        assert "doc2" not in result  # Should exclude overlapping docs
        assert len(result) <= 2  # Only doc3 and doc4 are hard negatives

    def test_sample_single_negative(self):
        """Test sampling with single negative document."""
        positive = ["doc1"]
        negative_pool = [["doc2"]]
        result = sample_hard_negatives(positive, negative_pool, num_negatives=5)
        assert len(result) <= 1
        assert "doc2" in result
