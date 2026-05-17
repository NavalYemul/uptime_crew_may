"""Shared fixtures for Day 4 tests."""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def mock_llm_fn():
    """Simple mock LLM function: returns 'mock response: {last message}'."""
    def _fn(messages):
        last = messages[-1].content if messages else "?"
        return f"Mock response to: {last[:50]}"
    return _fn


@pytest.fixture
def memory_checkpointer():
    """Fresh InMemorySaver for each test."""
    return MemorySaver()


@pytest.fixture
def human_message():
    return HumanMessage(content="Hello, this is a test message.")


@pytest.fixture
def sample_messages():
    return [
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="2+2 = 4"),
        HumanMessage(content="What about 3+3?"),
    ]
