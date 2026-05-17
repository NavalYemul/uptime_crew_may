import pytest
from langchain_core.messages import HumanMessage, AIMessage
from day5.multi_agent_orchestrator import (
    build_orchestrator, run_orchestrator,
    build_supervisor_orchestrator, OrchestratorState
)


class TestQueryRewrite:
    def test_query_is_rewritten(self, mock_retrieve_fn):
        g = build_orchestrator(
            rewrite_fn=lambda q: q + " REWRITTEN",
            retrieve_fn=mock_retrieve_fn
        )
        result = run_orchestrator(g, "hybrid search")
        assert result["rewritten_query"] == "hybrid search REWRITTEN"

    def test_default_rewrite_adds_suffix(self, mock_retrieve_fn):
        g = build_orchestrator(retrieve_fn=mock_retrieve_fn)
        result = run_orchestrator(g, "BM25")
        assert result["rewritten_query"] is not None
        assert len(result["rewritten_query"]) > len("BM25")


class TestRetrievalAgent:
    def test_retrieval_returns_docs(self, mock_retrieve_fn):
        g = build_orchestrator(retrieve_fn=mock_retrieve_fn)
        result = run_orchestrator(g, "hybrid search")
        assert len(result["retrieved_docs"]) > 0

    def test_empty_retrieval_routes_to_api(self):
        g = build_orchestrator(
            retrieve_fn=lambda q: [],  # always empty
            api_fn=lambda q: f"API: {q}"
        )
        result = run_orchestrator(g, "unknown query xyz")
        assert result["api_result"] is not None


class TestSynthesisAgent:
    def test_synthesis_present(self, mock_retrieve_fn, mock_synthesize_fn):
        g = build_orchestrator(
            retrieve_fn=mock_retrieve_fn,
            synthesize_fn=mock_synthesize_fn
        )
        result = run_orchestrator(g, "hybrid search")
        assert result["synthesis"] is not None
        assert len(result["synthesis"]) > 0

    def test_cost_tokens_tracked(self, mock_retrieve_fn):
        g = build_orchestrator(retrieve_fn=mock_retrieve_fn)
        result = run_orchestrator(g, "BM25 ranking algorithm")
        assert result["cost_tokens"] >= 0

    def test_messages_contain_ai_reply(self, mock_retrieve_fn):
        g = build_orchestrator(retrieve_fn=mock_retrieve_fn)
        result = run_orchestrator(g, "hybrid search")
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1


class TestSupervisor:
    def test_supervisor_runs_all_agents(self):
        agents = [
            ("agent_a", lambda t: f"A answer: {t}"),
            ("agent_b", lambda t: f"B answer: {t} with more detail"),
        ]
        g = build_supervisor_orchestrator(agents)
        result = g.invoke({"task": "test query", "agent_results": [], "final_answer": ""})
        assert len(result["agent_results"]) == 2
        assert result["final_answer"] != ""

    def test_supervisor_handles_agent_error(self):
        def bad_agent(t):
            raise Exception("fail")

        agents = [
            ("good", lambda t: "good answer"),
            ("bad",  bad_agent),
        ]
        # even with one broken agent, should complete
        g = build_supervisor_orchestrator(agents)
        result = g.invoke({"task": "query", "agent_results": [], "final_answer": ""})
        assert result["final_answer"] != ""
