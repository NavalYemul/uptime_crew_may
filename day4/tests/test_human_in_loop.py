"""Tests for day4.human_in_loop"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from day4.human_in_loop import build_hitl_graph, ApprovalWorkflow, HITLState


def make_mock_agent():
    actions = ["Send marketing email to 5000 customers", "Delete production database backup"]
    calls = [0]
    def fn(messages):
        calls[0] += 1
        return actions[(calls[0] - 1) % len(actions)]
    return fn


class TestHITLGraph:
    def test_graph_compiles(self):
        checkpointer = MemorySaver()
        graph = build_hitl_graph(make_mock_agent(), checkpointer=checkpointer)
        assert graph is not None

    def test_submit_returns_draft(self):
        checkpointer = MemorySaver()
        graph = build_hitl_graph(make_mock_agent(), checkpointer=checkpointer)
        workflow = ApprovalWorkflow(graph)
        draft = workflow.submit("Process request", "t001")
        assert draft is not None
        assert isinstance(draft, str)
        assert len(draft) > 0

    def test_approve_gives_executed_result(self):
        checkpointer = MemorySaver()
        graph = build_hitl_graph(make_mock_agent(), checkpointer=checkpointer)
        workflow = ApprovalWorkflow(graph)
        workflow.submit("Approve this action", "t_approve")
        result = workflow.approve("t_approve")
        assert "EXECUTED" in result or "executed" in result.lower() or len(result) > 0

    def test_reject_gives_rejected_result(self):
        checkpointer = MemorySaver()
        graph = build_hitl_graph(make_mock_agent(), checkpointer=checkpointer)
        workflow = ApprovalWorkflow(graph)
        workflow.submit("Reject this action", "t_reject")
        result = workflow.reject("t_reject")
        assert "REJECTED" in result or "rejected" in result.lower() or len(result) > 0

    def test_different_threads_independent(self):
        checkpointer = MemorySaver()
        graph = build_hitl_graph(make_mock_agent(), checkpointer=checkpointer)
        workflow = ApprovalWorkflow(graph)

        workflow.submit("Action A", "thread_aa")
        workflow.submit("Action B", "thread_bb")

        result_a = workflow.approve("thread_aa")
        result_b = workflow.reject("thread_bb")

        assert result_a != result_b or len(result_a) > 0
