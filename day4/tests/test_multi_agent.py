"""Tests for day4.multi_agent"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from day4.multi_agent import (
    build_research_subgraph, build_writer_subgraph,
    build_supervisor_graph, build_research_team,
    ResearchState, WriterState, SupervisorState
)


class TestResearchSubgraph:
    def test_subgraph_runs(self):
        def mock_research(topic): return f"Findings about {topic}"
        graph = build_research_subgraph(mock_research)
        result = graph.invoke({"messages": [], "topic": "AI trends", "findings": None})
        assert result["findings"] == "Findings about AI trends"

    def test_subgraph_adds_message(self):
        graph = build_research_subgraph(lambda t: "facts")
        result = graph.invoke({"messages": [], "topic": "test", "findings": None})
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][-1], AIMessage)


class TestWriterSubgraph:
    def test_writer_subgraph_runs(self):
        def mock_write(topic, findings): return f"Article about {topic}: {findings[:20]}"
        graph = build_writer_subgraph(mock_write)
        result = graph.invoke({"messages": [], "topic": "AI", "findings": "some facts", "draft": None})
        assert "Article about AI" in result["draft"]

    def test_draft_is_string(self):
        graph = build_writer_subgraph(lambda t, f: "Draft content")
        result = graph.invoke({"messages": [], "topic": "test", "findings": "f", "draft": None})
        assert isinstance(result["draft"], str)


class TestSupervisorGraph:
    def test_full_pipeline_runs(self):
        team = build_research_team()
        result = team.invoke({
            "messages":     [HumanMessage("Research AI")],
            "task":         "AI market research",
            "next_agent":   None,
            "research":     None,
            "draft":        None,
            "final_output": None,
        })
        assert result["research"] is not None
        assert result["draft"] is not None
        assert result["final_output"] is not None

    def test_final_output_contains_task(self):
        team = build_research_team()
        result = team.invoke({
            "messages":     [],
            "task":         "climate tech",
            "next_agent":   None,
            "research":     None,
            "draft":        None,
            "final_output": None,
        })
        assert "climate tech" in result["final_output"].lower()

    def test_messages_accumulate(self):
        team = build_research_team()
        result = team.invoke({
            "messages":     [HumanMessage("Start task")],
            "task":         "test task",
            "next_agent":   None,
            "research":     None,
            "draft":        None,
            "final_output": None,
        })
        assert len(result["messages"]) > 3  # supervisor + researcher + writer + finalise
