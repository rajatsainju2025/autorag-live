"""Tests for optimized string matching using trie data structures."""

import pytest

from autorag_live.utils.string_matching import DocumentMatcher, FastStringMatcher, SuffixTrie, Trie


class TestTrie:
    """Test basic trie implementation."""

    def test_basic_operations(self):
        """Test basic insert, search, and prefix operations."""
        trie = Trie()

        # Insert words
        trie.insert("hello")
        trie.insert("help")
        trie.insert("world")

        # Test search
        assert trie.search("hello")
        assert trie.search("help")
        assert trie.search("world")
        assert not trie.search("hel")
        assert not trie.search("nonexistent")

        # Test prefix matching
        assert trie.starts_with("hel")
        assert trie.starts_with("w")
        assert not trie.starts_with("xyz")

        # Test size
        assert len(trie) == 3

    def test_find_all_with_prefix(self):
        """Test finding all words with a given prefix."""
        trie = Trie()
        words = ["hello", "help", "helper", "world", "work", "worker"]

        for word in words:
            trie.insert(word)

        # Test prefix matching
        hel_words = trie.find_all_with_prefix("hel")
        assert set(hel_words) == {"hello", "help", "helper"}

        wor_words = trie.find_all_with_prefix("wor")
        assert set(wor_words) == {"world", "work", "worker"}

        # Test empty prefix
        all_words = trie.find_all_with_prefix("")
        assert set(all_words) == set(words)

    def test_remove_word(self):
        """Test removing words from the trie."""
        trie = Trie()
        trie.insert("hello")
        trie.insert("help")
        trie.insert("helper")

        assert len(trie) == 3

        # Remove a word
        assert trie.remove("help")
        assert not trie.search("help")
        assert trie.search("hello")
        assert trie.search("helper")
        assert len(trie) == 2

        # Try to remove non-existent word
        assert not trie.remove("nonexistent")
        assert len(trie) == 2

    def test_word_frequency(self):
        """Test word frequency tracking."""
        trie = Trie()
        trie.insert("hello")
        trie.insert("help")
        trie.insert("helper")

        # Test frequency counting
        assert trie.get_word_frequency("hel") == 3  # "hello", "help", "helper"
        assert trie.get_word_frequency("help") == 2  # "help", "helper"
        assert trie.get_word_frequency("hello") == 1
        assert trie.get_word_frequency("xyz") == 0

    def test_contains_operator(self):
        """Test 'in' operator support."""
        trie = Trie()
        trie.insert("test")

        assert "test" in trie
        assert "nonexistent" not in trie

    def test_case_insensitive(self):
        """Test case insensitive operations."""
        trie = Trie()
        trie.insert("Hello")
        trie.insert("WORLD")

        assert trie.search("hello")
        assert trie.search("world")
        assert trie.search("Hello")
        assert trie.search("WORLD")

    def test_empty_string(self):
        """Test handling of empty strings."""
        trie = Trie()
        trie.insert("")
        trie.insert("hello")

        assert not trie.search("")
        assert trie.search("hello")
        assert trie.starts_with("")


class TestSuffixTrie:
    """Test suffix trie implementation."""

    def test_suffix_operations(self):
        """Test suffix-specific operations."""
        suffix_trie = SuffixTrie()

        # Insert words
        suffix_trie.insert("hello")
        suffix_trie.insert("world")
        suffix_trie.insert("help")

        # Test suffix matching
        assert suffix_trie.ends_with("lo")  # hello
        assert suffix_trie.ends_with("ld")  # world
        assert suffix_trie.ends_with("lp")  # help
        assert not suffix_trie.ends_with("xyz")

    def test_find_all_with_suffix(self):
        """Test finding all words with a given suffix."""
        suffix_trie = SuffixTrie()
        words = ["hello", "yellow", "fellow", "world", "old"]

        for word in words:
            suffix_trie.insert(word)

        # Test suffix matching
        ow_words = suffix_trie.find_all_with_suffix("ow")
        assert set(ow_words) == {"yellow", "fellow"}

        llo_words = suffix_trie.find_all_with_suffix("llo")
        assert set(llo_words) == {"hello"}  # Only hello ends with 'llo'

        ld_words = suffix_trie.find_all_with_suffix("ld")
        assert set(ld_words) == {"world", "old"}

    def test_search_operation(self):
        """Test search in suffix trie."""
        suffix_trie = SuffixTrie()
        suffix_trie.insert("hello")
        suffix_trie.insert("world")

        assert suffix_trie.search("hello")
        assert suffix_trie.search("world")
        assert not suffix_trie.search("hel")


class TestFastStringMatcher:
    """Test comprehensive string matcher."""

    def test_basic_matching(self):
        """Test basic string matching operations."""
        matcher = FastStringMatcher()
        words = ["hello", "world", "help", "wonderful", "helper"]
        matcher.add_words(words)

        # Test prefix matching
        hel_matches = matcher.find_prefix_matches("hel")
        assert set(hel_matches) == {"hello", "help", "helper"}

        # Test suffix matching
        ful_matches = matcher.find_suffix_matches("ful")
        assert set(ful_matches) == {"wonderful"}

        # Test substring matching
        or_matches = matcher.find_substring_matches("or")
        assert set(or_matches) == {"world"}  # Only 'world' contains 'or'

    def test_pattern_matching(self):
        """Test complex pattern matching."""
        matcher = FastStringMatcher()
        words = ["hello", "help", "helpful", "world", "wonderful", "worker"]
        matcher.add_words(words)

        # Test combined constraints
        matches = matcher.find_pattern_matches(prefix="hel", suffix="ul")
        assert set(matches) == {"helpful"}

        matches = matcher.find_pattern_matches(prefix="wor", contains="r")
        assert set(matches) == {"world", "worker"}  # Both contain 'r' and start with 'wor'

        # Test no constraints (should return all words)
        all_matches = matcher.find_pattern_matches()
        assert set(all_matches) == set(words)

    def test_fuzzy_search(self):
        """Test fuzzy string matching."""
        matcher = FastStringMatcher()
        words = ["hello", "help", "world", "wonderful"]
        matcher.add_words(words)

        # Test fuzzy matching
        matches = matcher.fuzzy_search("helo", max_distance=1)
        match_words = [word for word, _ in matches]
        assert "hello" in match_words

        matches = matcher.fuzzy_search("halp", max_distance=2)
        match_words = [word for word, _ in matches]
        assert "help" in match_words

        # Test distance sorting
        matches = matcher.fuzzy_search("hel", max_distance=2)
        assert len(matches) >= 2
        # Distances should be sorted
        distances = [dist for _, dist in matches]
        assert distances == sorted(distances)

    def test_word_management(self):
        """Test adding and removing words."""
        matcher = FastStringMatcher()

        # Add words
        matcher.add_word("hello")
        matcher.add_word("world")
        assert len(matcher) == 2

        # Test membership
        assert "hello" in matcher
        assert "world" in matcher
        assert "nonexistent" not in matcher

        # Remove word
        assert matcher.remove_word("hello")
        assert len(matcher) == 1
        assert "hello" not in matcher

        # Try to remove non-existent word
        assert not matcher.remove_word("nonexistent")

        # Clear all
        matcher.clear()
        assert len(matcher) == 0

    def test_statistics(self):
        """Test statistics reporting."""
        matcher = FastStringMatcher()
        words = ["hello", "world", "help"]
        matcher.add_words(words)

        stats = matcher.get_statistics()
        assert stats["total_words"] == 3
        assert stats["prefix_trie_size"] == 3
        assert stats["suffix_trie_size"] == 3

    def test_duplicate_words(self):
        """Test handling of duplicate words."""
        matcher = FastStringMatcher()
        matcher.add_word("hello")
        matcher.add_word("hello")  # Duplicate

        assert len(matcher) == 1
        assert "hello" in matcher


class TestDocumentMatcher:
    """Test document-level string matching."""

    def test_document_processing(self):
        """Test adding and processing documents."""
        documents = [
            "Hello world, this is a test document",
            "Help me find the right answer",
            "World of possibilities awaits",
        ]

        matcher = DocumentMatcher(documents)

        assert matcher.get_document_count() == 3
        assert matcher.get_word_count() > 0

    def test_find_documents_with_prefix(self):
        """Test finding documents containing words with prefix."""
        documents = [
            "Hello world",
            "Help needed",
            "World peace",
            "Nothing here",
        ]

        matcher = DocumentMatcher(documents)

        # Find documents with words starting with "hel"
        hel_docs = matcher.find_documents_with_prefix("hel")
        assert len(hel_docs) == 2  # "Hello world" and "Help needed"

        # Find documents with words starting with "wor"
        wor_docs = matcher.find_documents_with_prefix("wor")
        assert len(wor_docs) == 2  # "Hello world" and "World peace"

        # Find documents with non-existent prefix
        xyz_docs = matcher.find_documents_with_prefix("xyz")
        assert len(xyz_docs) == 0

    def test_find_documents_with_pattern(self):
        """Test finding documents with complex patterns."""
        documents = [
            "The helpful person",
            "Beautiful world",
            "Wonderful day",
            "Simple test",
        ]

        matcher = DocumentMatcher(documents)

        # Find documents with words ending in "ful"
        ful_docs = matcher.find_documents_with_pattern(suffix="ful")
        assert len(ful_docs) == 3  # helpful, beautiful, wonderful

        # Find documents with words starting with "b"
        b_docs = matcher.find_documents_with_pattern(prefix="b")
        assert len(b_docs) == 1  # beautiful

        # Complex pattern
        pattern_docs = matcher.find_documents_with_pattern(prefix="w", suffix="ful")
        assert len(pattern_docs) == 1  # wonderful

    def test_empty_documents(self):
        """Test handling of empty documents."""
        matcher = DocumentMatcher()
        matcher.add_document("")
        matcher.add_document("   ")  # Whitespace only
        matcher.add_document("Hello world")

        assert matcher.get_document_count() == 3
        docs = matcher.find_documents_with_prefix("hel")
        assert len(docs) == 1


class TestPerformanceComparison:
    """Performance comparison tests (for manual verification)."""

    @pytest.mark.slow
    def test_prefix_search_performance(self):
        """Compare prefix search performance (manual verification)."""
        import time

        # Create large word list
        words = [f"word_{i:06d}" for i in range(10000)]
        words.extend([f"test_{i:06d}" for i in range(5000)])

        # Test linear search (naive approach)
        def linear_prefix_search(word_list, prefix):
            return [word for word in word_list if word.startswith(prefix)]

        # Time linear search
        start = time.perf_counter()
        linear_results = linear_prefix_search(words, "test_")
        linear_time = time.perf_counter() - start

        # Time trie-based search
        trie = Trie()
        for word in words:
            trie.insert(word)

        start = time.perf_counter()
        trie_results = trie.find_all_with_prefix("test_")
        trie_time = time.perf_counter() - start

        # Results should be the same
        assert set(linear_results) == set(trie_results)

        # Print timing results for manual verification
        print("\nPrefix Search Performance (15K words):")
        print(f"Linear search: {linear_time:.4f}s")
        print(f"Trie search: {trie_time:.4f}s")
        if linear_time > 0:
            print(f"Speedup: {linear_time/trie_time:.2f}x")

    @pytest.mark.slow
    def test_document_matching_performance(self):
        """Compare document matching performance (manual verification)."""
        import time

        # Create large document set
        documents = []
        for i in range(1000):
            doc = f"Document {i} contains words like test_{i} and helper_{i}"
            documents.append(doc)

        # Naive approach: linear search through all documents
        def naive_document_search(docs, prefix):
            matching_docs = []
            for doc in docs:
                words = doc.lower().split()
                if any(word.startswith(prefix) for word in words):
                    matching_docs.append(doc)
            return matching_docs

        # Time naive approach
        start = time.perf_counter()
        naive_results = naive_document_search(documents, "test_")
        naive_time = time.perf_counter() - start

        # Time optimized approach
        matcher = DocumentMatcher(documents)

        start = time.perf_counter()
        optimized_results = matcher.find_documents_with_prefix("test_")
        optimized_time = time.perf_counter() - start

        # Results should be the same (allowing for tokenization differences)
        assert len(naive_results) == len(optimized_results)

        # Print timing results for manual verification
        print("\nDocument Matching Performance (1K docs):")
        print(f"Naive search: {naive_time:.4f}s")
        print(f"Optimized search: {optimized_time:.4f}s")
        if naive_time > 0:
            print(f"Speedup: {naive_time/optimized_time:.2f}x")
