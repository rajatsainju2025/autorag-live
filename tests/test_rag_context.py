"""Tests for RAGContext — the unified pipeline state object."""

from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument


class TestRAGContextCreation:
    """Test context creation and factory methods."""

    def test_create_minimal(self):
        ctx = RAGContext.create(query="What is RLHF?")
        assert ctx.query == "What is RLHF?"
        assert ctx.current_stage == ContextStage.CREATED
        assert ctx.document_count == 0
        assert ctx.answer is None

    def test_create_with_metadata(self):
        ctx = RAGContext.create(
            query="test", metadata={"source": "api"}, tags=frozenset(["urgent"])
        )
        assert ctx.metadata["source"] == "api"
        assert "urgent" in ctx.tags


class TestRAGContextMutations:
    """Test copy-on-write mutation methods."""

    def test_add_documents_immutable(self):
        ctx = RAGContext.create(query="test")
        doc = RetrievedDocument(doc_id="d1", content="hello", score=0.9)
        ctx2 = ctx.add_documents([doc])

        assert ctx.document_count == 0  # original unchanged
        assert ctx2.document_count == 1
        assert ctx2.documents[0].doc_id == "d1"

    def test_add_reasoning_trace(self):
        ctx = RAGContext.create(query="test")
        ctx = ctx.add_reasoning_trace("Retrieved 2 docs", stage="retrieval")
        ctx = ctx.add_reasoning_trace("Reranked by relevance", stage="reranking")

        assert len(ctx.reasoning_traces) == 2
        assert ctx.reasoning_traces[0].step == 1
        assert ctx.reasoning_traces[1].step == 2

    def test_with_answer(self):
        ctx = RAGContext.create(query="test")
        ctx = ctx.with_answer("RLHF is reinforcement learning from human feedback", confidence=0.95)
        assert ctx.answer is not None
        assert ctx.confidence == 0.95

    def test_confidence_clamped(self):
        ctx = RAGContext.create(query="test").with_answer("x", confidence=1.5)
        assert ctx.confidence == 1.0

    def test_advance_stage(self):
        ctx = RAGContext.create(query="test")
        ctx = ctx.advance_stage(ContextStage.RETRIEVAL)
        assert ctx.current_stage == ContextStage.RETRIEVAL

    def test_add_eval_score(self):
        ctx = RAGContext.create(query="test")
        ctx = ctx.add_eval_score("faithfulness", 0.85, details="good grounding")
        ctx = ctx.add_eval_score("relevance", 0.9)
        assert len(ctx.eval_scores) == 2
        assert abs(ctx.avg_eval_score - 0.875) < 1e-6

    def test_record_latency(self):
        ctx = RAGContext.create(query="test")
        ctx = ctx.record_latency(ContextStage.RETRIEVAL, 120.5)
        ctx = ctx.record_latency(ContextStage.GENERATION, 350.0)
        assert abs(ctx.total_latency_ms - 470.5) < 1e-6

    def test_mark_error(self):
        ctx = RAGContext.create(query="test")
        ctx = ctx.mark_error("Timeout in retrieval")
        assert ctx.has_error
        assert ctx.metadata["error"] == "Timeout in retrieval"

    def test_add_tag(self):
        ctx = RAGContext.create(query="test")
        ctx = ctx.add_tag("high_priority")
        assert "high_priority" in ctx.tags

    def test_replace_documents(self):
        doc1 = RetrievedDocument(doc_id="d1", content="a", score=0.5)
        doc2 = RetrievedDocument(doc_id="d2", content="b", score=0.9)
        ctx = RAGContext.create(query="test").add_documents([doc1])
        ctx = ctx.replace_documents([doc2])
        assert ctx.document_count == 1
        assert ctx.documents[0].doc_id == "d2"


class TestRAGContextSerialisation:
    """Test JSON serialisation round-trip."""

    def test_round_trip(self):
        ctx = RAGContext.create(query="What is RLHF?", tags=frozenset(["ml"]))
        ctx = ctx.add_documents([RetrievedDocument(doc_id="d1", content="hello", score=0.9)])
        ctx = ctx.add_reasoning_trace("step one")
        ctx = ctx.with_answer("An answer", confidence=0.8)
        ctx = ctx.add_eval_score("faithfulness", 0.9)
        ctx = ctx.record_latency(ContextStage.RETRIEVAL, 100.0)

        data = ctx.to_dict()
        restored = RAGContext.from_dict(data)

        assert restored.query == ctx.query
        assert restored.document_count == 1
        assert restored.answer == "An answer"
        assert len(restored.reasoning_traces) == 1
        assert len(restored.eval_scores) == 1

    def test_to_json(self):
        ctx = RAGContext.create(query="test")
        json_str = ctx.to_json()
        assert '"query": "test"' in json_str


class TestRAGContextProperties:
    """Test computed properties."""

    def test_top_documents(self):
        docs = [
            RetrievedDocument(doc_id=f"d{i}", content=f"doc {i}", score=i * 0.1) for i in range(10)
        ]
        ctx = RAGContext.create(query="test").add_documents(docs)
        top = ctx.top_documents
        assert len(top) == 5
        assert top[0].score >= top[1].score  # sorted descending

    def test_is_complete(self):
        ctx = RAGContext.create(query="test")
        assert not ctx.is_complete
        ctx = ctx.advance_stage(ContextStage.COMPLETE)
        assert ctx.is_complete

    def test_repr(self):
        ctx = RAGContext.create(query="What is RLHF?")
        r = repr(ctx)
        assert "RAGContext" in r
        assert "RLHF" in r
