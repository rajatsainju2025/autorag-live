"""Unit tests for routing modules."""

import numpy as np

from autorag_live.routing.intent_classifier import IntentClassifier


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    async def embed_documents(self, texts):
        """Return random embeddings."""
        return [np.random.rand(100) for _ in texts]

    async def embed_query(self, text):
        """Return random embedding."""
        return np.random.rand(100)


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    def test_init(self):
        """Test initialization."""
        classifier = IntentClassifier()
        assert classifier is not None

    def test_classify_basic(self):
        """Test basic classification."""
        classifier = IntentClassifier()
        intent = classifier.classify("what is the capital of france")
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_classify_empty(self):
        """Test with empty text."""
        classifier = IntentClassifier()
        intent = classifier.classify("")
        assert isinstance(intent, str)

    def test_classify_multiple(self):
        """Test classifying multiple texts."""
        classifier = IntentClassifier()
        texts = ["explain foo", "what is bar", "how do I baz"]
        intents = [classifier.classify(t) for t in texts]
        assert len(intents) == 3


class TestRoutingModules:
    """Tests for routing module functionality."""

    def test_semantic_router_with_mock_model(self):
        """Test semantic router with mocked embedding model."""
        from autorag_live.routing.semantic_router import Route, SemanticRouter

        model = MockEmbeddingModel()
        router = SemanticRouter(embedding_model=model, similarity_threshold=0.7)

        # Add some routes
        router.add_route(
            Route(
                name="retrieval",
                utterances=["find documents", "search for information"],
            )
        )
        router.add_route(Route(name="qa", utterances=["answer the question", "what is"]))

        assert len(router.routes) == 2

    def test_intelligent_router_basic(self):
        """Test intelligent router."""
        from autorag_live.routing.intelligent_router import IntelligentRouter

        router = IntelligentRouter()
        assert router is not None

    def test_load_balancer(self):
        """Test load balancer."""
        from autorag_live.routing.load_balancer import LoadBalancer

        balancer = LoadBalancer(strategy="round_robin")
        assert balancer is not None

    def test_intent_classifier_patterns(self):
        """Test intent classification on different patterns."""
        classifier = IntentClassifier()

        queries = [
            "what is python",
            "how do I learn ML",
            "explain this concept",
            "search for documents",
        ]

        for query in queries:
            intent = classifier.classify(query)
            assert isinstance(intent, str)
