"""Tests for day4.memory_persistence"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from day4.memory_persistence import (
    build_chatbot_graph, chat_turn, get_conversation_history,
    get_state_history
)


def make_mock_llm():
    counter = [0]
    def fn(messages):
        counter[0] += 1
        last = messages[-1].content if messages else "?"
        return f"Reply #{counter[0]} to: {last[:30]}"
    return fn


class TestChatbotGraph:
    def test_stateless_responds(self):
        graph = build_chatbot_graph(make_mock_llm())
        result = graph.invoke({"messages": [HumanMessage("Hello")]})
        assert len(result["messages"]) >= 2
        last = result["messages"][-1]
        assert isinstance(last, AIMessage)

    def test_with_checkpointer_preserves_history(self):
        checkpointer = MemorySaver()
        llm = make_mock_llm()
        graph = build_chatbot_graph(llm, checkpointer=checkpointer)

        chat_turn(graph, "My name is Naval.", "test_thread")
        chat_turn(graph, "What is LangGraph?", "test_thread")

        history = get_conversation_history(graph, "test_thread")
        assert len(history) == 4  # 2 human + 2 ai messages

    def test_different_threads_isolated(self):
        checkpointer = MemorySaver()
        graph = build_chatbot_graph(make_mock_llm(), checkpointer=checkpointer)

        chat_turn(graph, "Thread A message", "thread_a")
        chat_turn(graph, "Thread B message", "thread_b")

        history_a = get_conversation_history(graph, "thread_a")
        history_b = get_conversation_history(graph, "thread_b")

        assert len(history_a) == 2
        assert len(history_b) == 2
        assert history_a[0]["content"] != history_b[0]["content"]

    def test_chat_turn_returns_string(self):
        checkpointer = MemorySaver()
        graph = build_chatbot_graph(make_mock_llm(), checkpointer=checkpointer)
        reply = chat_turn(graph, "Hello!", "thread_test")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_history_has_correct_roles(self):
        checkpointer = MemorySaver()
        graph = build_chatbot_graph(make_mock_llm(), checkpointer=checkpointer)
        chat_turn(graph, "Question 1", "role_thread")

        history = get_conversation_history(graph, "role_thread")
        roles = [h["role"] for h in history]
        assert "human" in roles
        assert "ai" in roles

    def test_state_history_has_snapshots(self):
        checkpointer = MemorySaver()
        graph = build_chatbot_graph(make_mock_llm(), checkpointer=checkpointer)

        chat_turn(graph, "Turn 1", "snapshot_thread")
        chat_turn(graph, "Turn 2", "snapshot_thread")

        snapshots = get_state_history(graph, "snapshot_thread")
        assert len(snapshots) >= 2
