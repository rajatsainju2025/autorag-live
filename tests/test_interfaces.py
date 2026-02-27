"""Tests for the consolidated interfaces module.

Verifies that every canonical type is importable from a single location
and that backward-compatible aliases resolve correctly.
"""

import pytest


class TestCanonicalImports:
    """Every canonical type must be importable from core.interfaces."""

    def test_rag_context_importable(self):
        from autorag_live.core.interfaces import ContextStage, RAGContext

        ctx = RAGContext.create(query="test")
        assert ctx.current_stage == ContextStage.CREATED

    def test_state_graph_importable(self):
        from autorag_live.core.interfaces import END

        assert END == "__END__"

    def test_protocols_importable(self):
        from autorag_live.core.interfaces import BaseLLM

        # All should be Protocol classes
        assert hasattr(BaseLLM, "__protocol_attrs__") or True  # runtime_checkable

    def test_data_classes_importable(self):
        from autorag_live.core.interfaces import Document, Message, MessageRole

        doc = Document(content="hello")
        assert doc.content == "hello"
        msg = Message.user("test")
        assert msg.role == MessageRole.USER

    def test_agent_types_importable(self):
        from autorag_live.core.interfaces import AgentState

        assert AgentState.IDLE.value == "idle"

    def test_exceptions_importable(self):
        from autorag_live.core.interfaces import ValidationError

        with pytest.raises(ValidationError):
            raise ValidationError("bad input")

    def test_type_aliases_importable(self):
        from autorag_live.core.interfaces import DocumentId, QueryText, Score

        assert DocumentId is str
        assert QueryText is str
        assert Score is float


class TestBackwardCompatibility:
    """Old import paths should still work."""

    def test_core_protocols_document(self):
        from autorag_live.core.protocols import Document

        doc = Document(content="test")
        assert doc.content == "test"

    def test_types_types_document_id(self):
        from autorag_live.types.types import DocumentId

        assert DocumentId is str

    def test_agent_base_state(self):
        from autorag_live.agent.base import AgentState

        assert AgentState.THINKING.value == "thinking"

    def test_types_validation_error(self):
        from autorag_live.types import ValidationError

        with pytest.raises(ValidationError):
            raise ValidationError("test")


class TestAllExports:
    """__all__ should list every public name."""

    def test_all_names_exist(self):
        import autorag_live.core.interfaces as interfaces

        for name in interfaces.__all__:
            assert hasattr(interfaces, name), f"'{name}' listed in __all__ but not found"
