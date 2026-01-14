"""
Integration tests for modular agentic RAG components.

This module provides basic smoke tests for all new modular agentic RAG modules
to verify that imports work and basic instantiation is functional.
"""

import pytest

# =============================================================================
# Test Module Imports
# =============================================================================


class TestModuleImports:
    """Test that all new modules can be imported."""

    def test_modular_base_imports(self):
        """Test modular base module imports."""
        from autorag_live.agent.modular_base import AgentExecutionState, AgentLifecycleState

        assert AgentLifecycleState.CREATED is not None
        assert AgentExecutionState.IDLE is not None

    def test_tool_registry_imports(self):
        """Test tool registry module imports."""
        from autorag_live.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        assert registry is not None

    def test_advanced_memory_imports(self):
        """Test advanced memory module imports."""
        from autorag_live.agent.advanced_memory import WorkingMemory

        wm = WorkingMemory(capacity=10)
        assert wm.size == 0

    def test_advanced_reasoning_imports(self):
        """Test advanced reasoning module imports."""
        from autorag_live.agent.advanced_reasoning import ThoughtNode

        node = ThoughtNode(id="test", content="test thought", depth=0)
        assert node.is_leaf() is True

    def test_advanced_strategies_imports(self):
        """Test advanced retrieval strategies module imports."""
        from autorag_live.retrieval.advanced_strategies import RetrievalStrategy

        # RetrievalStrategy uses auto() enum values
        assert RetrievalStrategy.HYBRID is not None
        assert RetrievalStrategy.ITERATIVE is not None
        assert RetrievalStrategy.COLBERT is not None

    def test_advanced_orchestration_imports(self):
        """Test multi-agent orchestration module imports."""
        from autorag_live.multi_agent.advanced_orchestration import AgentRole

        # AgentRole uses auto() enum values
        assert AgentRole.RESEARCHER is not None
        assert AgentRole.CRITIC is not None
        assert AgentRole.GENERATOR is not None

    def test_agent_evaluation_imports(self):
        """Test agent evaluation module imports."""
        from autorag_live.evals.agent_evaluation import EvaluationDimension

        assert EvaluationDimension.ACCURACY.value == "accuracy"

    def test_async_pipeline_imports(self):
        """Test async pipeline module imports."""
        from autorag_live.pipeline.async_pipeline import CircuitBreaker, CircuitState

        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_advanced_config_imports(self):
        """Test advanced configuration module imports."""
        from autorag_live.config.advanced_config import ConfigurationManager

        manager = ConfigurationManager()
        manager.set("test.key", "value")
        assert manager.get("test.key") == "value"


# =============================================================================
# Test Basic Functionality
# =============================================================================


class TestBasicFunctionality:
    """Test basic functionality of new modules."""

    def test_working_memory_operations(self):
        """Test working memory basic operations."""
        from autorag_live.agent.advanced_memory import MemoryItem, MemoryType, WorkingMemory

        wm = WorkingMemory(capacity=3)
        for i in range(5):
            item = MemoryItem(
                id=f"item{i}",
                content=f"Content {i}",
                memory_type=MemoryType.WORKING,
            )
            wm.add(item)

        # Should maintain capacity
        assert wm.size == 3

    def test_tool_registry_decoration(self):
        """Test tool registry decorator."""
        from autorag_live.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(description="Double a number")
        def double(x: int) -> int:
            return x * 2

        schema = registry.get_schema("double")
        assert schema.description == "Double a number"

    def test_di_container_basic(self):
        """Test DI container basic registration."""
        from autorag_live.config.advanced_config import DIContainer, ServiceLifetime

        container = DIContainer()

        class TestService:
            def __init__(self):
                self.name = "test"

        container.register(TestService, TestService, ServiceLifetime.SINGLETON)
        instance = container.resolve(TestService)
        assert instance.name == "test"

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts closed."""
        from autorag_live.pipeline.async_pipeline import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=5)
        assert cb.state == CircuitState.CLOSED

    def test_pipeline_context(self):
        """Test pipeline context operations."""
        from autorag_live.pipeline.async_pipeline import PipelineContext

        ctx = PipelineContext(query="test query")
        ctx.set_result("stage1", {"key": "value"})
        assert ctx.get_result("stage1")["key"] == "value"

    def test_thought_node_tree(self):
        """Test thought node tree structure."""
        from autorag_live.agent.advanced_reasoning import ThoughtNode

        root = ThoughtNode(id="root", content="Root thought", depth=0)
        child = ThoughtNode(id="child", content="Child thought", depth=1, parent_id="root")

        assert root.is_root() is True
        assert child.is_root() is False

    def test_config_presets(self):
        """Test configuration presets."""
        from autorag_live.config.advanced_config import ConfigPresets

        dev = ConfigPresets.development()
        prod = ConfigPresets.production()

        assert dev["agent.verbose"] is True
        assert prod["reranker.enabled"] is True

    def test_evaluation_metric(self):
        """Test evaluation metric creation."""
        from autorag_live.evals.agent_evaluation import EvaluationDimension, EvaluationMetric

        metric = EvaluationMetric(
            name="test_accuracy",
            value=0.95,
            dimension=EvaluationDimension.ACCURACY,
        )
        assert metric.value == 0.95

    def test_agent_message(self):
        """Test agent message creation."""
        from autorag_live.multi_agent.advanced_orchestration import AgentMessage

        msg = AgentMessage(
            sender="agent1",
            content="Hello",
            message_type="greeting",
        )
        assert msg.sender == "agent1"

    def test_retrieval_result(self):
        """Test retrieval result creation."""
        from autorag_live.retrieval.advanced_strategies import (
            RetrievalResult,
            RetrievalStrategy,
            RetrievedDocument,
        )

        doc1 = RetrievedDocument(id="1", content="Doc 1", score=0.9)
        doc2 = RetrievedDocument(id="2", content="Doc 2", score=0.8)
        result = RetrievalResult(
            query="test query",
            documents=[doc1, doc2],
            strategy=RetrievalStrategy.SINGLE,
        )
        assert len(result.documents) == 2


# =============================================================================
# Test Async Operations
# =============================================================================


class TestAsyncOperations:
    """Test async operations in new modules."""

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test async tool execution."""
        from autorag_live.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(description="Multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        result = await registry.execute("multiply", a=3, b=4)
        assert result.success is True
        assert result.result == 12

    @pytest.mark.asyncio
    async def test_circuit_breaker_operations(self):
        """Test circuit breaker async operations."""
        from autorag_live.pipeline.async_pipeline import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2)

        assert await cb.can_execute() is True
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_event_emitter(self):
        """Test event emitter."""
        from autorag_live.pipeline.async_pipeline import (
            EventEmitter,
            PipelineEvent,
            PipelineEventType,
        )

        emitter = EventEmitter()
        received = []

        async def handler(event):
            received.append(event)

        emitter.on(PipelineEventType.STAGE_STARTED, handler)

        await emitter.emit(
            PipelineEvent(
                event_type=PipelineEventType.STAGE_STARTED,
                stage_name="test",
                data={},
            )
        )

        assert len(received) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
