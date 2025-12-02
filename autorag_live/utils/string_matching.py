"""
High-performance string matching using trie data structures.

This module provides efficient prefix/suffix matching and string search
capabilities that significantly outperform linear search for large datasets.
"""

from typing import Dict, List, Optional, Set, Tuple


class TrieNode:
    """A node in the trie data structure."""

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end_of_word: bool = False
        self.word_count: int = 0  # For frequency tracking
        self.value: Optional[str] = None  # Store the complete word


class Trie:
    """
    High-performance trie (prefix tree) for fast string operations.

    Provides O(m) time complexity for search, insert, and prefix operations,
    where m is the length of the string being processed.

    Example:
        >>> trie = Trie()
        >>> trie.insert("hello")
        >>> trie.insert("help")
        >>> trie.search("hello")  # True
        >>> trie.starts_with("hel")  # True
        >>> trie.find_all_with_prefix("hel")  # ["hello", "help"]
    """

    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, word: str) -> None:
        """Insert a word into the trie."""
        if not word:
            return

        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.word_count += 1

        if not node.is_end_of_word:
            node.is_end_of_word = True
            node.value = word
            self.size += 1

    def search(self, word: str) -> bool:
        """Check if a word exists in the trie."""
        if not word:
            return False

        node = self._find_node(word.lower())
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word in the trie starts with the given prefix."""
        if not prefix:
            return self.size > 0

        return self._find_node(prefix.lower()) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find the node corresponding to the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def find_all_with_prefix(self, prefix: str) -> List[str]:
        """Find all words that start with the given prefix."""
        if not prefix:
            return self.get_all_words()

        node = self._find_node(prefix.lower())
        if node is None:
            return []

        words = []
        self._collect_words(node, words)
        return words

    def _collect_words(self, node: TrieNode, words: List[str]) -> None:
        """Recursively collect all words from a given node."""
        if node.is_end_of_word and node.value:
            words.append(node.value)

        for child in node.children.values():
            self._collect_words(child, words)

    def get_all_words(self) -> List[str]:
        """Get all words stored in the trie."""
        words = []
        self._collect_words(self.root, words)
        return words

    def remove(self, word: str) -> bool:
        """Remove a word from the trie. Returns True if word was removed."""
        if not word or not self.search(word):
            return False

        self._remove_helper(self.root, word.lower(), 0)
        self.size -= 1
        return True

    def _remove_helper(self, node: TrieNode, word: str, index: int) -> bool:
        """Helper method for removing words."""
        if index == len(word):
            # We've reached the end of the word
            if not node.is_end_of_word:
                return False

            node.is_end_of_word = False
            node.value = None
            node.word_count -= 1

            # If node has no children, it can be deleted
            return len(node.children) == 0 and node.word_count == 0

        char = word[index]
        if char not in node.children:
            return False

        child_node = node.children[char]
        should_delete_child = self._remove_helper(child_node, word, index + 1)

        if should_delete_child:
            del node.children[char]
            node.word_count -= 1

        # Return True if current node should be deleted
        return not node.is_end_of_word and len(node.children) == 0 and node.word_count == 0

    def get_word_frequency(self, prefix: str) -> int:
        """Get the frequency of words with the given prefix."""
        node = self._find_node(prefix.lower())
        return node.word_count if node else 0

    def __len__(self) -> int:
        """Return the number of words in the trie."""
        return self.size

    def __contains__(self, word: str) -> bool:
        """Support 'in' operator for membership testing."""
        return self.search(word)


class SuffixTrie(Trie):
    """
    Trie optimized for suffix matching operations.

    Stores words in reverse order to enable fast suffix lookups.
    """

    def insert(self, word: str) -> None:
        """Insert a word (reversed) into the suffix trie."""
        if word:
            super().insert(word[::-1])

    def search(self, word: str) -> bool:
        """Check if a word exists in the suffix trie."""
        if word:
            return super().search(word[::-1])
        return False

    def ends_with(self, suffix: str) -> bool:
        """Check if any word ends with the given suffix."""
        if suffix:
            return self.starts_with(suffix[::-1])
        return self.size > 0

    def find_all_with_suffix(self, suffix: str) -> List[str]:
        """Find all words that end with the given suffix."""
        if not suffix:
            return self.get_all_words()

        reversed_words = super().find_all_with_prefix(suffix[::-1])
        return [word[::-1] for word in reversed_words]

    def get_all_words(self) -> List[str]:
        """Get all words stored in the suffix trie (unreversed)."""
        reversed_words = super().get_all_words()
        return [word[::-1] for word in reversed_words]


class FastStringMatcher:
    """
    High-performance string matching using multiple trie structures.

    Combines prefix and suffix tries for comprehensive string matching
    capabilities with O(m) time complexity.
    """

    def __init__(self):
        self.prefix_trie = Trie()
        self.suffix_trie = SuffixTrie()
        self.words: Set[str] = set()

    def add_words(self, words: List[str]) -> None:
        """Add multiple words to the matcher."""
        for word in words:
            self.add_word(word)

    def add_word(self, word: str) -> None:
        """Add a single word to the matcher."""
        if not word or word in self.words:
            return

        self.words.add(word)
        self.prefix_trie.insert(word)
        self.suffix_trie.insert(word)

    def find_prefix_matches(self, prefix: str) -> List[str]:
        """Find all words starting with the given prefix."""
        return self.prefix_trie.find_all_with_prefix(prefix)

    def find_suffix_matches(self, suffix: str) -> List[str]:
        """Find all words ending with the given suffix."""
        return self.suffix_trie.find_all_with_suffix(suffix)

    def find_substring_matches(self, substring: str) -> List[str]:
        """Find all words containing the given substring (slower operation)."""
        if not substring:
            return list(self.words)

        matches = []
        substring_lower = substring.lower()

        for word in self.words:
            if substring_lower in word.lower():
                matches.append(word)

        return matches

    def find_pattern_matches(
        self, prefix: str = "", suffix: str = "", contains: str = ""
    ) -> List[str]:
        """
        Find words matching multiple criteria.

        Args:
            prefix: Required prefix (empty string means no constraint)
            suffix: Required suffix (empty string means no constraint)
            contains: Required substring (empty string means no constraint)

        Returns:
            List of words matching all specified criteria
        """
        # Start with all words if no constraints
        if not prefix and not suffix and not contains:
            return list(self.words)

        # Get initial candidates from the most restrictive constraint
        candidates = set(self.words)

        if prefix:
            prefix_matches = self.find_prefix_matches(prefix)
            candidates &= set(prefix_matches)

        if suffix:
            suffix_matches = self.find_suffix_matches(suffix)
            candidates &= set(suffix_matches)

        if contains:
            substring_matches = self.find_substring_matches(contains)
            candidates &= set(substring_matches)

        return list(candidates)

    def fuzzy_search(self, query: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """
        Find words similar to the query using Levenshtein distance.

        Args:
            query: Query string
            max_distance: Maximum allowed edit distance

        Returns:
            List of (word, distance) tuples sorted by distance
        """
        if not query:
            return []

        matches = []
        query_lower = query.lower()

        for word in self.words:
            distance = self._levenshtein_distance(query_lower, word.lower())
            if distance <= max_distance:
                matches.append((word, distance))

        # Sort by distance (closest matches first)
        matches.sort(key=lambda x: x[1])
        return matches

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the matcher."""
        return {
            "total_words": len(self.words),
            "prefix_trie_size": len(self.prefix_trie),
            "suffix_trie_size": len(self.suffix_trie),
        }

    def remove_word(self, word: str) -> bool:
        """Remove a word from the matcher."""
        if word not in self.words:
            return False

        self.words.discard(word)
        self.prefix_trie.remove(word)
        self.suffix_trie.remove(word)
        return True

    def clear(self) -> None:
        """Remove all words from the matcher."""
        self.words.clear()
        self.prefix_trie = Trie()
        self.suffix_trie = SuffixTrie()

    def __len__(self) -> int:
        """Return the number of words in the matcher."""
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        """Support 'in' operator for membership testing."""
        return word in self.words


class DocumentMatcher:
    """
    Optimized string matching for document processing.

    Uses multiple string matching strategies optimized for document
    search and retrieval operations.
    """

    def __init__(self, documents: Optional[List[str]] = None):
        self.matcher = FastStringMatcher()
        self.doc_to_words: Dict[str, Set[str]] = {}

        if documents:
            self.add_documents(documents)

    def add_documents(self, documents: List[str]) -> None:
        """Add multiple documents to the matcher."""
        for doc in documents:
            self.add_document(doc)

    def add_document(self, document: str) -> None:
        """
        Add a document to the matcher.

        Tokenizes the document and adds all unique words to the trie structures.
        """
        if not document or not document.strip():
            # Still add empty documents to maintain count, but with empty word set
            self.doc_to_words[document] = set()
            return

        # Simple tokenization (can be enhanced)
        words = self._tokenize(document)
        unique_words = set(words)

        self.doc_to_words[document] = unique_words
        self.matcher.add_words(list(unique_words))

    def find_documents_with_prefix(self, prefix: str) -> List[str]:
        """Find documents containing words with the given prefix."""
        matching_words = set(self.matcher.find_prefix_matches(prefix))
        matching_docs = []

        for doc, words in self.doc_to_words.items():
            if words & matching_words:  # Set intersection
                matching_docs.append(doc)

        return matching_docs

    def find_documents_with_pattern(
        self, prefix: str = "", suffix: str = "", contains: str = ""
    ) -> List[str]:
        """Find documents containing words matching the pattern."""
        matching_words = set(
            self.matcher.find_pattern_matches(prefix=prefix, suffix=suffix, contains=contains)
        )
        matching_docs = []

        for doc, words in self.doc_to_words.items():
            if words & matching_words:
                matching_docs.append(doc)

        return matching_docs

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization (can be enhanced with more sophisticated methods)."""
        # Split on whitespace and basic punctuation
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def get_word_count(self) -> int:
        """Get the total number of unique words."""
        return len(self.matcher)

    def get_document_count(self) -> int:
        """Get the number of documents."""
        return len(self.doc_to_words)
