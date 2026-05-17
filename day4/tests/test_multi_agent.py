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


class TestAgenticRAG:
    def _make_docs(self):
        return ["LangGraph is a framework for building stateful agents",
                "Python is a programming language used in data science"]

    def test_agentic_rag_compiles(self):
        from day4.multi_agent import build_agentic_rag_graph
        graph = build_agentic_rag_graph(
            retrieve_fn=lambda q: ["doc1"],
            generate_fn=lambda q, d: "answer",
        )
        assert graph is not None

    def test_agentic_rag_with_relevant_docs_returns_answer(self):
        from day4.multi_agent import build_agentic_rag_graph
        docs = self._make_docs()
        graph = build_agentic_rag_graph(
            retrieve_fn=lambda q: [d for d in docs if any(w in d.lower() for w in q.lower().split())],
            generate_fn=lambda q, d: f"Answer based on {len(d)} docs",
        )
        result = graph.invoke({
            "query": "langgraph", "documents": [], "answer": "",
            "needs_rewrite": False, "iteration": 0,
        })
        assert result["answer"] != ""
        assert "docs" in result["answer"]

    def test_agentic_rag_with_empty_docs_rewrites_query(self):
        from day4.multi_agent import build_agentic_rag_graph
        rewrite_count = [0]

        def retrieve(q):
            # Return empty on first call, docs on subsequent calls
            if "detailed explanation" in q:
                rewrite_count[0] += 1
                return ["found document after rewrite"]
            return []

        graph = build_agentic_rag_graph(
            retrieve_fn=retrieve,
            generate_fn=lambda q, d: f"Generated from {len(d)} docs",
        )
        result = graph.invoke({
            "query": "unknown topic xyz", "documents": [], "answer": "",
            "needs_rewrite": False, "iteration": 0,
        })
        assert rewrite_count[0] > 0

    def test_agentic_rag_answer_key_present(self):
        from day4.multi_agent import build_agentic_rag_graph
        graph = build_agentic_rag_graph(
            retrieve_fn=lambda q: ["doc"],
            generate_fn=lambda q, d: "final answer",
        )
        result = graph.invoke({
            "query": "test", "documents": [], "answer": "",
            "needs_rewrite": False, "iteration": 0,
        })
        assert "answer" in result
        assert result["answer"] == "final answer"


class TestAgentTypeEnum:
    def test_enum_values(self):
        from day4.multi_agent import AgentType
        assert AgentType.REACT.value == "react"
        assert AgentType.PLAN_AND_EXECUTE.value == "plan_and_execute"
        assert AgentType.AGENTIC_RAG.value == "agentic_rag"

    def test_enum_members(self):
        from day4.multi_agent import AgentType
        assert len(AgentType) == 3
