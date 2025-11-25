"""Tests for string interning utilities."""
import pytest

from autorag_live.utils.string_interning import (
    QueryStringInterner,
    StringInterner,
    intern_query,
    intern_string,
)


def test_string_interner_basic():
    """Test basic string interning functionality."""
    interner = StringInterner()

    # Test same content returns same object
    s1 = interner.intern("hello")
    s2 = interner.intern("hello")
    assert s1 is s2  # Same identity

    # Test different content returns different objects
    s3 = interner.intern("world")
    assert s1 is not s3
    assert s1 != s3


def test_string_interner_stats():
    """Test string interner statistics."""
    interner = StringInterner()

    # Initial stats
    stats = interner.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["cache_size"] == 0

    # First intern is a miss
    interner.intern("test")
    stats = interner.get_stats()
    assert stats["misses"] == 1
    assert stats["cache_size"] == 1

    # Second intern is a hit
    interner.intern("test")
    stats = interner.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_string_interner_clear():
    """Test clearing the interner cache."""
    interner = StringInterner()

    interner.intern("test1")
    interner.intern("test2")

    stats = interner.get_stats()
    assert stats["cache_size"] == 2

    interner.clear()
    stats = interner.get_stats()
    assert stats["cache_size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_string_interner_type_validation():
    """Test that interner only accepts strings."""
    interner = StringInterner()

    with pytest.raises(TypeError):
        interner.intern(123)

    with pytest.raises(TypeError):
        interner.intern(None)

    with pytest.raises(TypeError):
        interner.intern(["hello"])


def test_query_string_interner():
    """Test query-specific string interning."""
    interner = QueryStringInterner()

    # Test query term interning
    query1 = "find machine learning algorithms"
    query2 = "find machine learning algorithms"

    result1 = interner.intern_query_terms(query1)
    result2 = interner.intern_query_terms(query2)

    # Should get same interned terms
    assert result1 == result2

    # Terms should be lowercased
    query3 = "FIND MACHINE LEARNING"
    result3 = interner.intern_query_terms(query3)
    assert result3 == "find machine learning"


def test_query_string_interner_empty():
    """Test query interner with empty/whitespace queries."""
    interner = QueryStringInterner()

    assert interner.intern_query_terms("") == ""
    assert interner.intern_query_terms("   ") == "   "
    assert interner.intern_query_terms(None) is None


def test_global_intern_string():
    """Test global string interning function."""
    s1 = intern_string("global_test")
    s2 = intern_string("global_test")
    assert s1 is s2


def test_global_intern_query():
    """Test global query interning function."""
    q1 = intern_query("search for documents")
    q2 = intern_query("search for documents")
    assert q1 == q2
    assert q1 == "search for documents"  # Should be lowercase


def test_query_interner_common_terms():
    """Test that common terms are pre-interned."""
    interner = QueryStringInterner()

    # These should be hits since they're pre-interned
    result = interner.intern_query_terms("the quick search")

    stats_after = interner.get_stats()

    # Should have more hits than misses due to common terms
    assert result == "the quick search"
    assert stats_after["hits"] > 0
