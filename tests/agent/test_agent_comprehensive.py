"""
Comprehensive test suite for agent components.

Tests agent behavior, multi-step reasoning, and streaming.
"""

import pytest

from autorag_live.agent import (
    Agent,
    AgenticRAGPipeline,
    ConversationMemory,
    ErrorRecoveryEngine,
    Reasoner,
    ToolRegistry,
)


class TestAgent:
    """Test basic agent functionality."""

    def test_agent_creation(self):
        """Test agent initialization."""
        agent = Agent(name="TestAgent", max_steps=5)
        assert agent.name == "TestAgent"
        assert agent.max_steps == 5
        assert len(agent.tools) == 0

    def test_agent_state_transitions(self):
        """Test agent state transitions."""
        agent = Agent()

        # Initial state
        assert agent.state.value == "idle"

        # Plan a goal
        plan = agent.plan("What is the capital of France?")
        assert agent.state.value == "thinking"
        assert len(plan) > 0

    def test_tool_registration(self):
        """Test tool registration."""
        agent = Agent()

        def dummy_tool(query: str) -> str:
            return f"Result for {query}"

        agent.register_tool("dummy", dummy_tool)
        assert "dummy" in agent.tools
        assert callable(agent.tools["dummy"])

    def test_agent_memory(self):
        """Test agent memory."""
        agent = Agent()
        agent.memory.add_message("user", "Hello")
        agent.memory.add_message("agent", "Hi there!")

        assert len(agent.memory.messages) == 2
        assert agent.memory.messages[0].role == "user"
        assert agent.memory.messages[1].role == "agent"


class TestReasoning:
    """Test reasoning and planning."""

    def test_reasoner_creation(self):
        """Test reasoner initialization."""
        reasoner = Reasoner(verbose=True)
        assert reasoner.verbose is True
        assert len(reasoner.reasoning_traces) == 0

    def test_reasoning_about_query(self):
        """Test query reasoning."""
        reasoner = Reasoner()
        trace = reasoner.reason_about_query("What is machine learning?")

        assert trace.query == "What is machine learning?"
        assert len(trace.steps) > 0
        assert trace.success is True
        assert trace.final_action is not None

    def test_action_plan_generation(self):
        """Test action plan generation."""
        reasoner = Reasoner()
        plan = reasoner.generate_action_plan("How does neural networks work?")

        assert plan["query"] == "How does neural networks work?"
        assert "reasoning_trace" in plan
        assert "action_sequence" in plan

    def test_goal_decomposition(self):
        """Test goal decomposition."""
        reasoner = Reasoner()
        subgoals = reasoner._decompose_goal("How can I learn Python?", "learn_process", {})

        assert len(subgoals) > 0
        assert isinstance(subgoals, list)
        assert all(isinstance(sg, str) for sg in subgoals)


class TestConversationMemory:
    """Test conversation memory management."""

    def test_memory_creation(self):
        """Test memory initialization."""
        memory = ConversationMemory(max_context_tokens=2048)
        assert memory.max_context_tokens == 2048
        assert len(memory.messages) == 0

    def test_add_messages(self):
        """Test adding messages."""
        memory = ConversationMemory()
        msg1 = memory.add_message("user", "Hello")
        msg2 = memory.add_message("assistant", "Hi")

        assert len(memory.messages) == 2
        assert msg1.role == "user"
        assert msg2.role == "assistant"

    def test_context_retrieval(self):
        """Test context window retrieval."""
        memory = ConversationMemory(max_context_tokens=1000)
        memory.add_message("user", "What is AI?")
        memory.add_message("assistant", "AI is artificial intelligence...")

        context = memory.get_context()
        assert "What is AI?" in context or "USER" in context.upper()

    def test_message_search(self):
        """Test message search by relevance."""
        memory = ConversationMemory()
        memory.add_message("user", "Tell me about machine learning")
        memory.add_message("assistant", "ML is a subset of AI")
        memory.add_message("user", "Can you explain neural networks?")

        results = memory.search_messages("neural networks", top_k=2)
        assert len(results) > 0
        assert isinstance(results[0][0].content, str)


class TestToolRegistry:
    """Test tool registry and execution."""

    def test_registry_creation(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        assert len(registry.tools) == 0
        assert len(registry.schemas) == 0

    def test_tool_registration(self):
        """Test tool registration."""
        registry = ToolRegistry()

        def search_tool(query: str, k: int = 5) -> list:
            return [f"Result {i} for {query}" for i in range(k)]

        schema = registry.register_tool("search", search_tool, description="Search tool")

        assert "search" in registry.tools
        assert schema.name == "search"
        assert len(schema.parameters) >= 2

    def test_tool_execution(self):
        """Test tool execution."""
        registry = ToolRegistry()

        def simple_tool(text: str) -> str:
            return text.upper()

        registry.register_tool("upper", simple_tool)

        result = registry.execute_tool("upper", {"text": "hello"})
        assert result == "HELLO"

    def test_tool_listing(self):
        """Test tool listing."""
        registry = ToolRegistry()

        def tool1() -> None:
            pass

        def tool2() -> None:
            pass

        registry.register_tool("tool1", tool1, tags=["test"])
        registry.register_tool("tool2", tool2, tags=["demo"])

        all_tools = registry.list_tools()
        assert len(all_tools) == 2

        test_tools = registry.list_tools(tag="test")
        assert len(test_tools) == 1


class TestRAGPipeline:
    """Test agentic RAG pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline initialization."""
        pipeline = AgenticRAGPipeline()
        assert pipeline.agent is not None
        assert pipeline.tool_registry is not None

    def test_query_processing(self):
        """Test query processing."""
        pipeline = AgenticRAGPipeline()
        response = pipeline.process_query("What is machine learning?")

        assert response.query == "What is machine learning?"
        assert response.answer is not None
        assert response.iterations > 0


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def test_retry_success(self):
        """Test successful retry."""
        engine = ErrorRecoveryEngine()
        call_count = [0]

        def flaky_operation():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary error")
            return "Success"

        result, success = engine.retry_with_backoff(flaky_operation)
        assert success is True
        assert result == "Success"
        assert call_count[0] == 3

    def test_fallback_chain(self):
        """Test fallback chain."""
        engine = ErrorRecoveryEngine()

        def failing_op():
            raise ValueError("Fails")

        def fallback_op():
            return "Fallback result"

        result, fallback_name, success = engine.try_fallback_chain(
            [("primary", failing_op), ("fallback", fallback_op)]
        )

        assert success is True
        assert result == "Fallback result"
        assert fallback_name == "fallback"


class TestStreaming:
    """Test streaming functionality."""

    def test_streaming_handler_creation(self):
        """Test streaming handler initialization."""
        from autorag_live.agent import StreamingResponseHandler

        handler = StreamingResponseHandler()
        assert handler is not None
        assert not handler.is_cancelled()

    def test_stream_events(self):
        """Test stream event generation."""
        from autorag_live.agent import StreamingResponseHandler
        from autorag_live.agent.streaming import StreamEventType

        handler = StreamingResponseHandler()
        handler.start_streaming()
        handler.add_token("Hello")
        handler.add_token(" World")
        handler.finish_streaming()

        events = handler.get_all_events()
        assert len(events) > 0
        assert events[0].event_type == StreamEventType.START


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
